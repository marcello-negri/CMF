"""Implementations of Cholesky related transformation:

- TransformDiagonalExponential(Transform):
  input: matrix
  output: matrix with exponential of diagonal elements (hence positive)

- TransformDiagonalSoftplus(Transform);
  input: matrix
  output: matrix with softmax of diagonal elements (hence positive)

- Triangular(Transform):
  input: lower triangular matrix
  output: lower triangular matrix times learnable triangular matrix

  note: the product of two lower triangular matrices is also lower triangular

- SumOfSigmoidsTriangular(SumOfSigmoids):
  input: lower triangular matrix
  output: element wise transformation through SumOfSigmoid

- ExtendedSoftplusTriangular(ExtendedSoftplus):
  helper function for SumOfSigmoidTriangular

- CholeskyOuterProduct(Transform):
  input: lower triangular matrix
  output: lower triangular times transpose of lower triangular

  note: if the lower triangular matrices has positive diagonal elements,
        the resulting matrix is symmetric positive-definite

- ActNormTriangular(Transform):
  input: lower triangular matrix
  output: lower triangular matrix after activation normalization

  note: normalization is initialized in a data dependent way and then learnt independently
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from nflows.distributions import Distribution
from nflows.transforms.base import Transform, CompositeTransform
from nflows.transforms.conditional import ConditionalTransform
from nflows.transforms.adaptive_sigmoids import SumOfSigmoids, ExtendedSoftplus
from nflows.transforms.nonlinearities import Exp, Sigmoid, Softplus
from nflows.transforms.linear import  ScalarScale, ScalarShift
from nflows.transforms.no_analytic_inv.base import MonotonicTransform
from torch.nn import init
from nflows.utils import torchutils

class StandardNormalTriangular(Distribution):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, features, N):
        super().__init__()
        self._shape = torch.Size(features)

        self.N = N
        self.lower_indices = np.tril_indices(self.N, k=0)

        self.register_buffer("_log_z",
                             torch.tensor(0.5 * np.prod(features) * np.log(2 * np.pi),
                                          dtype=torch.float64),
                             persistent=False)

    def _log_prob(self, inputs_tril, context):
        # Note: the context is ignored.
        inputs = inputs_tril[:, self.lower_indices[0], self.lower_indices[1]]
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * \
            torchutils.sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            samples = torch.randn(num_samples, *self._shape, device=self._log_z.device)
            samples_tril = torch.zeros(1, self.N, self.N, device=self._log_z.device)
            samples_tril[:, self.lower_indices[0], self.lower_indices[1]] = samples
            return samples_tril
        else:
            # The value of the context is ignored, only its size and device are taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape,
                                  device=context.device)
            samples = torchutils.split_leading_dim(samples, [context_size, num_samples])

            samples_tril = torch.zeros(context_size, num_samples, self.N, self.N, device=context.device)
            samples_tril[:, :, self.lower_indices[0], self.lower_indices[1]] = samples
            return samples_tril


class MaskTriangular(Transform):

    def __init__(self, N):
        super().__init__()
        self.N = N
        self.lower_indices = np.tril_indices(self.N, k=0)

        mask = torch.zeros(1, self.N, self.N)
        mask[:, self.lower_indices[0], self.lower_indices[1]] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def forward(self, inputs, context=None):
        outputs = self.mask * inputs

        logabsdet = inputs.new_zeros(inputs.shape[0])

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        raise NotImplementedError()

    def check_lower_triangular(self, inputs):
        assert torch.all(inputs[..., self.upper_indices[0], self.upper_indices[1]] == 0.), (
            'tensor must be lower triangular matrix')


fancy_exp_transform = CompositeTransform([Sigmoid(),
                                          ScalarScale(scale=80., trainable=False),
                                          Exp(),
                                          ScalarShift(1e-5, trainable=False)])

fancy_softplus_transform = CompositeTransform([Sigmoid(),
                                               ScalarScale(scale=80., trainable=False),
                                               Softplus(),
                                               ScalarShift(1e-5, trainable=False)])


class TransformDiagonal(Transform):
    def __init__(self, N, diag_transformation: Transform = Exp()):
        super().__init__()
        self.N = N
        self.diag_indices = np.diag_indices(self.N)
        self.diag_mask = nn.Parameter(torch.diag_embed(torch.ones(1, self.N)), requires_grad=False)
        self.diag_transform = diag_transformation

        # self.transform = CompositeTransform([Sigmoid(), ScalarScale(scale=self.MAX_EXP, trainable=False)])

    def forward(self, inputs, context=None):
        transformed_diag, logabsdet_diag = self.diag_transform(torch.diagonal(inputs, dim1=-2, dim2=-1))
        outputs = torch.diagonal_scatter(inputs, transformed_diag, dim1=-2, dim2=-1)
        return outputs, logabsdet_diag

    def inverse(self, inputs, context=None):
        transformed_diag, logabsdet_diag = self.diag_transform.inverse(torch.diagonal(inputs, dim1=-2, dim2=-1))
        outputs = torch.diagonal_scatter(inputs, transformed_diag, dim1=-2, dim2=-1)
        return outputs, logabsdet_diag


class TransformDiagonalExponential(TransformDiagonal):
    def __init__(self, N, eps=1e-5):
        super().__init__(N=N, diag_transformation=CompositeTransform([Exp(),
                                                                      ScalarShift(eps, trainable=False)]))


class TransformDiagonalSoftplus(TransformDiagonal):
    def __init__(self, N, eps=1e-5):
        super().__init__(N=N, diag_transformation=CompositeTransform([Softplus(),
                                                                      ScalarShift(eps, trainable=False)]))


class Triangular(Transform):

    def __init__(self, features, raw_params=None, identity_init=True):
        super().__init__()

        self.features = features
        self.N = check_lower_triangular(features)
        self.lower_indices = np.tril_indices(self.N, k=0)
        self.upper_indices = np.triu_indices(self.N, k=1)
        self.diag_indices = np.diag_indices(self.N)

        if raw_params is None:
            self._initialize_tril_bias(identity_init)
        else:
            self._initialize_tril_rawparams(raw_params)

        mask = self.tril.new_zeros(1, self.N, self.N)
        mask[:, self.lower_indices[0], self.lower_indices[1]] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

        # powers used to compute logabsdet
        self.powers = nn.Parameter(torch.arange(self.N, 0, -1, device=self.tril.device), requires_grad=False)


    def _initialize_tril_bias(self, identity_init):
        if identity_init:
            tril = torch.zeros(1, self.N, self.N)
            tril[:, self.diag_indices[0], self.diag_indices[1]] = 1.
        else:
            stdv = 1.0 / np.sqrt(self.features)
            tril = torch.rand(1, self.N, self.N) * 2 * stdv - stdv
            tril[:, self.upper_indices[0], self.upper_indices[1]] = 0.

        self.tril = nn.Parameter(tril, requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1, self.N, self.N), requires_grad=True)

    def _initialize_tril_rawparams(self, raw_params):
        assert raw_params.shape[1:] == (self.features, 2)

        tril_param = raw_params[:, :, 0]
        bias_param = raw_params[:, :, 1]

        self.tril = raw_params.new_zeros(raw_params.shape[0], self.N, self.N)
        self.tril[:, self.lower_indices[0], self.lower_indices[1]] = tril_param

        self.bias = raw_params.new_zeros(raw_params.shape[0], self.N, self.N)
        self.bias[:, self.lower_indices[0], self.lower_indices[1]] = bias_param


    def forward(self, inputs, context=None):
        self.check_lower_triangular(inputs)
        self.check_upper_not_learnt()

        outputs = inputs @ (self.tril * self.mask) + (self.bias * self.mask)

        tril_diagonal = self.tril[:, self.diag_indices[0], self.diag_indices[1]]
        logabsdet = (tril_diagonal.abs().log() * self.powers).sum(-1)
        #print(logabsdet.cpu().detach().numpy())
        #breakpoint()
        #logabsdet = logabsdet.repeat(inputs.shape[0])

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        self.check_lower_triangular(inputs)

        shifted_inputs = inputs - (self.bias * self.mask)
        outputs = torch.linalg.solve_triangular(self.tril * self.mask, shifted_inputs, left=False, upper=False)

        tril_diagonal = self.tril[:, self.diag_indices[0], self.diag_indices[1]]
        logabsdet = (tril_diagonal.abs().log() * self.powers).sum()
        logabsdet = - logabsdet.repeat(inputs.shape[0])

        return outputs, logabsdet

    def check_lower_triangular(self, inputs):
        assert torch.all(inputs[..., self.upper_indices[0], self.upper_indices[1]] == 0.), (
            'tensor must be lower triangular matrix')

    def check_upper_not_learnt(self):
        try:
            assert torch.all(self.tril[:, self.upper_indices[0], self.upper_indices[1]] == 0.)
            assert torch.all(self.bias[:, self.upper_indices[0], self.upper_indices[1]] == 0.)
        except:
            breakpoint()


class SumOfSigmoidsTriangular(SumOfSigmoids):
    PREACT_SCALE_MIN = .1
    PREACT_SCALE_MAX = 10.
    PREACT_SHIFT_MAX = 2

    def __init__(self, features, n_sigmoids=10, num_iterations=35, lim=80, raw_params: torch.Tensor=None):

        self.n_sigmoids = n_sigmoids
        self.features = features
        self.N = check_lower_triangular(features)
        self.indices = np.tril_indices(self.N)

        super(SumOfSigmoidsTriangular, self).__init__(features=features, n_sigmoids=n_sigmoids,
                                                      num_iterations=num_iterations, lim=lim, raw_params=raw_params)

        if raw_params is None:
            self.extended_softplus = ExtendedSoftplusTriangular(features=features)

            shift_preact = torch.zeros(1, self.N, self.N, self.n_sigmoids)
            log_scale_preact = torch.zeros(1, self.N, self.N, self.n_sigmoids)
            raw_softmax = torch.zeros(1, self.N, self.N, self.n_sigmoids)

            shift_preact[:, self.indices[0], self.indices[1], :] = self.shift_preact.detach().clone()
            log_scale_preact[:, self.indices[0], self.indices[1], :] = self.log_scale_preact.detach().clone()
            raw_softmax[:, self.indices[0], self.indices[1], :] = self.raw_softmax.detach().clone()

            self.shift_preact = nn.Parameter(shift_preact, requires_grad=True)
            self.log_scale_preact = nn.Parameter(log_scale_preact, requires_grad=True)
            self.raw_softmax = nn.Parameter(raw_softmax, requires_grad=False)

        else:
            assert raw_params.shape[1:] == (features, 3 * self.n_sigmoids + 1)
            self.set_raw_tril_params(features, raw_params)

        log_scale_postact = torch.zeros(self.shift_preact.shape[0], self.N, self.N, self.n_sigmoids, device=self.shift_preact.device)
        log_scale_postact[:, self.indices[0], self.indices[1], :] = self.log_scale_postact.detach().clone()
        self.log_scale_postact = nn.Parameter(log_scale_postact, requires_grad=False)

        mask = self.shift_preact.new_zeros(1, self.N, self.N, self.n_sigmoids)
        mask[:, self.indices[0], self.indices[1], :] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

        self.eps = 1e-4

    def forward(self, inputs, context=None):
        self.check_lower_triangular(inputs)
        self.check_upper_not_learnt()

        output_sum_of_sigmoids, log_diag_jac_sigmoids = self.sum_of_sigmoids(inputs)
        output_extended_softplus, log_diag_jac_esoftplus = self.extended_softplus(inputs)

        output = output_sum_of_sigmoids + output_extended_softplus
        logabsdet = torch.logaddexp(log_diag_jac_sigmoids, log_diag_jac_esoftplus).sum(-1)

        return output, logabsdet

    def set_raw_tril_params(self, features, raw_params):
        # 3 = shift, scale, softmax for sigmoids
        # 2 = log_scale, log_shift for extended softplus
        vals = torch.split(raw_params, [self.n_sigmoids, self.n_sigmoids, self.n_sigmoids, 1], dim=-1)
        shift_preact, log_scale_preact, raw_softmax = vals[:3]
        self.extended_softplus = ExtendedSoftplusTriangular(features=features, shift=vals[3])

        mb = raw_params.shape[0]
        self.shift_preact = shift_preact.new_zeros(mb, self.N, self.N, self.n_sigmoids)
        self.log_scale_preact = shift_preact.new_zeros(mb, self.N, self.N, self.n_sigmoids)
        self.raw_softmax = shift_preact.new_zeros(mb, self.N, self.N, self.n_sigmoids)

        self.shift_preact[:, self.indices[0], self.indices[1], :] = shift_preact
        self.log_scale_preact[:, self.indices[0], self.indices[1], :] = log_scale_preact
        self.raw_softmax[:, self.indices[0], self.indices[1], :] = raw_softmax

    def sum_of_sigmoids(self, inputs):
        shift_preact_tril, scale_preact_tril, scale_postact_tril = self.get_params()
        pre_act_tril = (scale_preact_tril * self.mask) * (inputs.unsqueeze(-1) - shift_preact_tril * self.mask)

        indices = np.tril_indices(inputs.shape[-1])
        pre_act_raveled = pre_act_tril[:, indices[0], indices[1], :]
        scale_postact_raveled = scale_postact_tril[:, indices[0], indices[1], :]
        scale_preact_raveled = scale_preact_tril[:, indices[0], indices[1], :]

        sigmoids_expanded = (scale_postact_tril * self.mask) * torch.sigmoid(pre_act_tril * self.mask)
        log_jac_sigmoid_expanded = scale_postact_raveled.log() + scale_preact_raveled.log() + self.sigmoid_log_derivative(pre_act_raveled)

        return sigmoids_expanded.sum(-1), torch.logsumexp(log_jac_sigmoid_expanded, -1)

    def check_lower_triangular(self, inputs):
        upper_indices = np.triu_indices(self.N, k=1)
        assert torch.all(inputs[:, upper_indices[0], upper_indices[1]]) == 0., (
            "input tensor must be mini batch of lower triangular matrices")

    def check_upper_not_learnt(self):
        try:
            upper_indices = np.triu_indices(self.N, k=1)
            assert torch.all(self.shift_preact[:, upper_indices[0], upper_indices[1], :] == 0.)
            assert torch.all(self.log_scale_preact[:, upper_indices[0], upper_indices[1], :] == 0.)
            assert torch.all(self.raw_softmax[:, upper_indices[0], upper_indices[1], :] == 0.)
            assert torch.all(self.log_scale_postact[:, upper_indices[0], upper_indices[1], :] == 0.)
        except:
            breakpoint()


class ExtendedSoftplusTriangular(ExtendedSoftplus):
    """
    Combination of a (shifted and scaled) softplus and the same softplus flipped around the origin
    Softplus(scale * (x-shift)) - Softplus(-scale * (x + shift))
    Linear outside of origin, flat around origin.
    """

    def __init__(self, features, shift=None, eps=1e-5):
        super().__init__(features, shift)
        self.eps = eps

        self.N = check_lower_triangular(features)
        self.indices = np.tril_indices(self.N)
        shift = self.shift.new_zeros(self.shift.shape[0], self.N, self.N)
        shift[:, self.indices[0], self.indices[1]] = self.shift.detach().clone()
        self.shift = nn.Parameter(shift, requires_grad=True)

        mask = self.shift.new_zeros(1, self.N, self.N)
        mask[:, self.indices[0], self.indices[1]] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)


    def forward(self, inputs):
        self.check_upper_not_learnt()

        shift_tril = self.get_shift()
        shift_tril_masked = shift_tril * self.mask
        outputs = self.softplus(inputs, shift_tril_masked) + self.softminus(inputs, shift_tril_masked)

        inputs_raveled = inputs[:, self.indices[0], self.indices[1]]
        shift_raveled = shift_tril[:, self.indices[0], self.indices[1]]

        diag_jacobian = torch.logaddexp(self.log_diag_jacobian_pos(inputs_raveled, shift_raveled),
                                        self.log_diag_jacobian_neg(inputs_raveled, shift_raveled))
        return outputs, diag_jacobian  # torch.log(diag_jacobian).sum(-1)

    def check_upper_not_learnt(self):
        try:
            upper_indices = np.triu_indices(self.N, k=1)
            assert torch.all(self.shift[:, upper_indices[0], upper_indices[1]] == 0.)
        except:
            breakpoint()


# class SumOfSigmoidsTriangular(MonotonicTransform):
#     PREACT_SCALE_MIN = .1
#     PREACT_SCALE_MAX = 10.
#     PREACT_SHIFT_MAX = 2
#
#     def __init__(self, features, n_sigmoids=10, num_iterations=35, lim=80, raw_params: torch.Tensor = None):
#
#         self.n_sigmoids = n_sigmoids
#         self.features = features
#         self.N = check_lower_triangular(features)
#         self.indices = np.tril_indices(self.N)
#
#         super(SumOfSigmoidsTriangular, self).__init__(num_iterations=num_iterations, lim=lim)
#         if raw_params is None:
#             self.initialize_tril_params()
#         else:
#             assert raw_params.shape[1:] == (features, 3 * self.n_sigmoids + 1)
#             self.set_raw_tril_params(features, raw_params)
#
#         log_scale_postact = self.shift_preact.new_zeros(self.shift_preact.shape[0], self.N, self.N, self.n_sigmoids)
#         self.log_scale_postact = nn.Parameter(log_scale_postact, requires_grad=False)
#
#         mask = self.shift_preact.new_zeros(1, self.N, self.N, self.n_sigmoids)
#         mask[:, self.indices[0], self.indices[1], :] = 1.
#         self.mask = nn.Parameter(mask, requires_grad=False)
#
#         self.eps = 1e-4
#
#     def forward(self, inputs, context=None):
#         self.check_lower_triangular(inputs)
#         self.check_upper_not_learnt()
#
#         output_sum_of_sigmoids, log_diag_jac_sigmoids = self.sum_of_sigmoids(inputs)
#         output_extended_softplus, log_diag_jac_esoftplus = self.extended_softplus(inputs)
#
#         output = output_sum_of_sigmoids + output_extended_softplus
#         logabsdet = torch.logaddexp(log_diag_jac_sigmoids, log_diag_jac_esoftplus).sum(-1)
#
#         return output, logabsdet
#
#     def set_raw_tril_params(self, features, raw_params):
#         # 3 = shift, scale, softmax for sigmoids
#         # 2 = log_scale, log_shift for extended softplus
#         vals = torch.split(raw_params, [self.n_sigmoids, self.n_sigmoids, self.n_sigmoids, 1], dim=-1)
#         shift_preact, log_scale_preact, raw_softmax = vals[:3]
#         self.extended_softplus = ExtendedSoftplusTriangular(features=features, given_shift=vals[3])
#
#         mb = raw_params.shape[0]
#         self.shift_preact = shift_preact.new_zeros(mb, self.N, self.N, self.n_sigmoids)
#         self.log_scale_preact = shift_preact.new_zeros(mb, self.N, self.N, self.n_sigmoids)
#         self.raw_softmax = shift_preact.new_zeros(mb, self.N, self.N, self.n_sigmoids)
#
#         self.shift_preact[:, self.indices[0], self.indices[1], :] = shift_preact
#         self.log_scale_preact[:, self.indices[0], self.indices[1], :] = log_scale_preact
#         self.raw_softmax[:, self.indices[0], self.indices[1], :] = raw_softmax
#
#     def initialize_tril_params(self):
#         self.extended_softplus = ExtendedSoftplusTriangular(features=self.features)
#
#         shift_preact_features = torch.randn(1, self.features, self.n_sigmoids)
#         log_scale_preact_features = torch.zeros(1, self.features, self.n_sigmoids)
#         raw_softmax_features = torch.ones(1, self.features, self.n_sigmoids)
#
#         shift_preact = torch.zeros(1, self.N, self.N, self.n_sigmoids)
#         log_scale_preact = torch.zeros(1, self.N, self.N, self.n_sigmoids)
#         raw_softmax = torch.zeros(1, self.N, self.N, self.n_sigmoids)
#
#         shift_preact[:, self.indices[0], self.indices[1], :] = shift_preact_features
#         log_scale_preact[:, self.indices[0], self.indices[1], :] = log_scale_preact_features
#         raw_softmax[:, self.indices[0], self.indices[1], :] = raw_softmax_features
#
#         self.shift_preact = nn.Parameter(shift_preact, requires_grad=True)
#         self.log_scale_preact = nn.Parameter(log_scale_preact, requires_grad=True)
#         self.raw_softmax = nn.Parameter(raw_softmax, requires_grad=False)
#
#     def sum_of_sigmoids(self, inputs):
#         shift_preact_tril, scale_preact_tril, scale_postact_tril = self.get_params()
#         pre_act_tril = (scale_preact_tril * self.mask) * (inputs.unsqueeze(-1) - shift_preact_tril * self.mask)
#
#         indices = np.tril_indices(inputs.shape[-1])
#         pre_act_raveled = pre_act_tril[:, indices[0], indices[1], :]
#         scale_postact_raveled = scale_postact_tril[:, indices[0], indices[1], :]
#         scale_preact_raveled = scale_preact_tril[:, indices[0], indices[1], :]
#
#         sigmoids_expanded = (scale_postact_tril * self.mask) * torch.sigmoid(pre_act_tril * self.mask)
#         log_jac_sigmoid_expanded = scale_postact_raveled.log() + scale_preact_raveled.log() + self.sigmoid_log_derivative(pre_act_raveled)
#
#         return sigmoids_expanded.sum(-1), torch.logsumexp(log_jac_sigmoid_expanded, -1)
#
#     def sigmoid_log_derivative(self, x):
#         return x - 2 * torch.nn.functional.softplus(x)
#
#     def get_params(self):
#         scale_postact = torch.exp(self.log_scale_postact * self.mask) * \
#                         torch.nn.functional.softmax(self.raw_softmax * self.mask, dim=-1) + self.eps
#         scale_preact = torch.sigmoid(self.log_scale_preact * self.mask)
#         scale_preact = scale_preact * (self.PREACT_SCALE_MAX - self.PREACT_SCALE_MIN) + self.PREACT_SCALE_MIN
#
#         shift_preact = torch.tanh(self.shift_preact * self.mask) * self.PREACT_SHIFT_MAX
#
#         return shift_preact, scale_preact, scale_postact
#
#     def check_lower_triangular(self, inputs):
#         upper_indices = np.triu_indices(self.N, k=1)
#         assert torch.all(inputs[:, upper_indices[0], upper_indices[1]]) == 0., (
#             "input tensor must be mini batch of lower triangular matrices")
#
#     def check_upper_not_learnt(self):
#         try:
#             upper_indices = np.triu_indices(self.N, k=1)
#             assert torch.all(self.shift_preact[:, upper_indices[0], upper_indices[1], :] == 0.)
#             assert torch.all(self.log_scale_preact[:, upper_indices[0], upper_indices[1], :] == 0.)
#             assert torch.all(self.raw_softmax[:, upper_indices[0], upper_indices[1], :] == 0.)
#             assert torch.all(self.log_scale_postact[:, upper_indices[0], upper_indices[1], :] == 0.)
#         except:
#             breakpoint()
#
#
# class ExtendedSoftplusTriangular(nn.Module):
#     """
#     Combination of a (shifted and scaled) softplus and the same softplus flipped around the origin
#
#     Softplus(scale * (x-shift)) - Softplus(-scale * (x + shift))
#
#     Linear outside of origin, flat around origin.
#     """
#
#     def __init__(self, features, given_shift=None, eps=1e-3):
#         self.features = features
#         self.eps = eps
#
#         super(ExtendedSoftplusTriangular, self).__init__()
#
#         self.N = check_lower_triangular(features)
#         self.indices = np.tril_indices(self.N)
#         if given_shift is None:
#             shift_features = torch.ones(1, features) * 3
#             # self.log_scale = torch.nn.Parameter(torch.zeros(1, features), requires_grad=True)
#         elif torch.is_tensor(given_shift):
#             shift_features = given_shift.reshape(-1, features)
#             # self.log_scale = log_scale.reshape(-1, features)
#         else:
#             shift_features = torch.tensor(given_shift)
#             # self.log_scale = torch.nn.Parameter(torch.tensor(log_scale), requires_grad=True)
#
#         self._softplus = torch.nn.Softplus()
#
#         shift = shift_features.new_zeros(shift_features.shape[0], self.N, self.N)
#         shift[:, self.indices[0], self.indices[1]] = shift_features
#         self.shift = nn.Parameter(shift, requires_grad=True)
#
#         mask = shift_features.new_zeros(1, self.N, self.N)
#         mask[:, self.indices[0], self.indices[1]] = 1.
#         self.mask = nn.Parameter(mask, requires_grad=False)
#
#     def get_shift(self):
#         return self._softplus(self.shift) + self.eps
#
#     def softplus(self, x, shift):
#         return self._softplus((x - shift))
#
#     def softminus(self, x, shift):
#         return - self._softplus(-(x + shift))
#
#     def diag_jacobian_pos(self, x, shift):
#         # (b e^(b x))/(e^(a b) + e^(b x))
#         return torch.exp(x) / (torch.exp(shift) + torch.exp(x))
#
#     def log_diag_jacobian_pos(self, x, shift):
#         # -log(e^(a b) + e^(b x)) + b x + log(b)
#         log_jac = -torch.logaddexp(shift, x) + x
#         return log_jac
#
#     def diag_jacobian_neg(self, x, shift):
#         return torch.sigmoid(- (shift + x))
#
#     def log_diag_jacobian_neg(self, x, shift):
#         return - self._softplus((shift + x))
#
#     def forward(self, inputs):
#         self.check_upper_not_learnt()
#
#         shift_tril = self.get_shift()
#         shift_tril_masked = shift_tril * self.mask
#         outputs = self.softplus(inputs, shift_tril_masked) + self.softminus(inputs, shift_tril_masked)
#
#         inputs_raveled = inputs[:, self.indices[0], self.indices[1]]
#         shift_raveled = shift_tril[:, self.indices[0], self.indices[1]]
#
#         diag_jacobian = torch.logaddexp(self.log_diag_jacobian_pos(inputs_raveled, shift_raveled),
#                                         self.log_diag_jacobian_neg(inputs_raveled, shift_raveled))
#         return outputs, diag_jacobian  # torch.log(diag_jacobian).sum(-1)
#
#     def check_upper_not_learnt(self):
#         try:
#             upper_indices = np.triu_indices(self.N, k=1)
#             assert torch.all(self.shift[:, upper_indices[0], upper_indices[1]] == 0.)
#         except:
#             breakpoint()


class CholeskyOuterProduct(Transform):
    def __init__(self, N, checkargs=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.N = N
        self.eye = nn.Parameter(torch.diag_embed(torch.ones(self.N)).unsqueeze(0), requires_grad=False)
        self.powers = nn.Parameter(torch.arange(self.N, 0, -1).unsqueeze(0), requires_grad=False)
        self.checkargs = checkargs

    def forward(self, inputs, context=None):
        if self.checkargs:
            self.check_pos_low_triang(inputs)
        outputs = torch.bmm(inputs, inputs.mT)
        outputs = 0.5*(outputs + outputs.mT)
        diagonal = torch.diagonal(inputs, dim1=-2, dim2=-1)
        logabsdet = self.N * np.log(2.) + (self.powers * diagonal.log()).sum(-1)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        inputs_jitter = inputs + self.eye * self.eps
        if self.checkargs:
            self.check_pos_def(inputs_jitter)

        outputs = torch.linalg.cholesky(inputs_jitter, upper=False)
        diagonal = torch.diagonal(outputs, dim1=-2, dim2=-1)
        logabsdet = self.N * np.log(2.) + (self.powers * diagonal.log()).sum(1)

        return outputs, -logabsdet

    def check_pos_low_triang(self, inputs):
        assert inputs.shape[-2] == inputs.shape[-1], "input tensor must be mini batch of square matrices"
        upper_indices = np.triu_indices(self.N, k=1)
        assert torch.all(inputs[:, upper_indices[0], upper_indices[1]] == 0.), (
            "input tensor must be mini batch of lower triangular matrices")
        assert torch.all(torch.diagonal(inputs, dim1=-2, dim2=-1) > 0), (
            'input tensor must be mini batch of lower triangular matrices with positive diagonal elements')

    def check_pos_def(self, inputs):
        assert torch.all(inputs == inputs.mT), "Input matrix is not symmetric."
        assert torch.all(torch.linalg.eig(inputs)[0].real >= 0), (
            "Input matrix is not positive semi-definite in order to perform Cholesky decomposition"
        )

class ActNormTriangular(Transform):
    def __init__(self, features, mu=None, std=None, raw_params: torch.Tensor=None):
        super().__init__()

        self.features = features
        self.N = check_lower_triangular(features)

        self.lower_indices = np.tril_indices(self.N, k=0)
        self.upper_indices = np.triu_indices(self.N, k=1)

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("raw_parameters", torch.tensor(False, dtype=torch.bool))

        if raw_params is None:
            self.log_scale = nn.Parameter(torch.zeros(1, self.N, self.N), requires_grad=True)
            self.shift = nn.Parameter(torch.zeros(1, self.N, self.N), requires_grad=True)
        else:
            mb = raw_params.shape[0]
            self.log_scale_param = self.constrained_raw_params(raw_params[:, :, 0])
            self.shift_param = self.constrained_raw_params(raw_params[:, :, 1])
            self.log_scale = raw_params.new_zeros(mb, self.N, self.N)
            self.shift = raw_params.new_zeros(mb, self.N, self.N)

            with torch.no_grad():
                self.std = std
                self.mu = mu

            self.raw_parameters.data = torch.tensor(True, dtype=torch.bool)

        mask = self.shift.new_zeros(1, self.N, self.N)
        mask[:, self.lower_indices[0], self.lower_indices[1]] = 1.
        self.mask = nn.Parameter(mask, requires_grad=False)

    def compute_mu_std(self, inputs):
        with torch.no_grad():
            lower_indices = (..., self.lower_indices[0], self.lower_indices[1])
            std = inputs[lower_indices].std(dim=0)
            mu = (inputs[lower_indices] / std).mean(dim=0)
        return mu, std

    def constrained_raw_params(self, inputs):
        return inputs #5 * (torch.sigmoid(inputs * 0.5) - 0.5)

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def forward(self, inputs, context=None):

        if self.training and not self.initialized:
            self._initialize(inputs)

        self.check_upper_not_learnt()

        one_like_scale = self.scale.new_ones(self.scale.shape, requires_grad=False)
        scale_masked = self.scale * self.mask + one_like_scale * (1. - self.mask)

        outputs = scale_masked * inputs + (self.shift * self.mask)

        log_scale_tril = self.log_scale[:, self.lower_indices[0], self.lower_indices[1]]

        if self.raw_parameters:
            logabsdet = log_scale_tril.sum(1)
        else:
            logabsdet = log_scale_tril.sum() * outputs.new_ones(inputs.shape[0])

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        self.check_upper_not_learnt()

        one_like_scale = self.scale.new_ones(self.scale.shape, requires_grad=False)
        scale_masked = self.scale * self.mask + one_like_scale * (1. - self.mask)
        outputs = (inputs - (self.shift * self.mask)) / scale_masked

        log_scale_tril = self.log_scale[:, self.lower_indices[0], self.lower_indices[1]]
        if self.raw_parameters:
            logabsdet = -log_scale_tril.sum(1)
        else:
            logabsdet = -log_scale_tril.sum() * outputs.new_ones(inputs.shape[0])

        return outputs, logabsdet

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        lower_indices = (..., self.lower_indices[0], self.lower_indices[1])
        mu, std = self.compute_mu_std(inputs)
        if self.raw_parameters:
            if self.std is None and self.mu is None:
                self.log_scale[lower_indices] = - torch.log(std) + self.log_scale_param
                self.shift[lower_indices] = - mu + self.shift_param
            elif self.std is not None and self.mu is not None:
                self.log_scale[lower_indices] = - torch.log(self.std) + self.log_scale_param
                self.shift[lower_indices] = - self.mu + self.shift_param
            else:
                raise ValueError("Both std and mu must be either None or a specified tensor")
        else:
            with torch.no_grad():
                self.log_scale.data[lower_indices] = - torch.log(std)
                self.shift.data[lower_indices] = - mu
        with torch.no_grad():
            self.initialized.data = torch.tensor(True, dtype=torch.bool)

    def check_upper_not_learnt(self):
        try:
            assert torch.all(self.log_scale[:, self.upper_indices[0], self.upper_indices[1]] == 0.)
            assert torch.all(self.shift[:, self.upper_indices[0], self.upper_indices[1]] == 0.)
        except:
            breakpoint()


class ConditionalSumOfSigmoidsTriangular(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            n_sigmoids=10,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        self.n_sigmoids = n_sigmoids
        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _output_dim_multiplier(self):
        return 3 * self.n_sigmoids + 1

    def _forward_given_params(self, inputs, autoregressive_params):
        transformer = SumOfSigmoidsTriangular(n_sigmoids=self.n_sigmoids, features=self.features,
                                    raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                          self._output_dim_multiplier()))
        z, logabsdet = transformer(inputs)
        return z, logabsdet

    def _inverse_given_params(self, inputs, autoregressive_params):
        transformer = SumOfSigmoidsTriangular(n_sigmoids=self.n_sigmoids, features=self.features,
                                    raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                          self._output_dim_multiplier()))
        x, logabsdet = transformer.inverse(inputs)
        return x, logabsdet


class ConditionalTriangular(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )

    def _output_dim_multiplier(self):
        return 2

    def _forward_given_params(self, inputs, autoregressive_params):
        transformer = Triangular(features=self.features,
                                 raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                       self._output_dim_multiplier()))
        z, logabsdet = transformer(inputs)
        return z, logabsdet

    def _inverse_given_params(self, inputs, autoregressive_params):
        transformer = Triangular(features=self.features,
                                 raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                       self._output_dim_multiplier()))
        x, logabsdet = transformer.inverse(inputs)
        return x, logabsdet


class ConditionalActNormTriangular(ConditionalTransform):
    def __init__(
            self,
            features,
            hidden_features,
            context_features=None,
            num_blocks=2,
            use_residual_blocks=True,
            activation=F.relu,
            dropout_probability=0.0,
            use_batch_norm=False,
    ):
        super().__init__(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            use_residual_blocks=use_residual_blocks,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm
        )
        self.register_buffer("init_mu_std", torch.tensor(False, dtype=torch.bool))

    def _output_dim_multiplier(self):
        return 2

    def _forward_given_params(self, inputs, autoregressive_params):
        if self.init_mu_std:
            transformer = ActNormTriangular(features=self.features, mu=self.mu, std=self.std,
                                     raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                           self._output_dim_multiplier()))
        else:
            transformer = ActNormTriangular(features=self.features,
                                            raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                                  self._output_dim_multiplier()))
            self.mu, self.std = transformer.compute_mu_std(inputs)
            self.init_mu_std.data = torch.tensor(True, dtype=torch.bool)

        z, logabsdet = transformer(inputs)
        return z, logabsdet

    def _inverse_given_params(self, inputs, autoregressive_params):
        if self.init_mu_std:
            transformer = ActNormTriangular(features=self.features, mu=self.mu, std=self.std,
                                            raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                                  self._output_dim_multiplier()))
        else:
            transformer = ActNormTriangular(features=self.features,
                                            raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                                  self._output_dim_multiplier()))
            self.mu, self.std = transformer.compute_mu_std(inputs)
            self.init_mu_std.data = torch.tensor(True, dtype=torch.bool)

        x, logabsdet = transformer.inverse(inputs)
        return x, logabsdet


def check_lower_triangular(p):
    assert p > 0, "dimension must be positive number"
    temp = 1 + 8 * p
    assert np.square(np.floor(np.sqrt(temp))) == temp, "invalid dimension: can't be mapped to lower triangular matrix"
    N = int((-1 + np.floor(np.sqrt(temp))) // 2)

    return N