"""Implementations of Cholesky related transformation:

- FillTriangular:
  input: vector of size p * (p + 1) / 2
  output: lower triangular matrix of size p x p filled in with input values

- FillSymmetricZeroDiag:
  input: vector of size p * (p - 1) / 2
  output: symmetric matrix with zero on the diagonal
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from nflows.transforms.base import Transform
from nflows.transforms.normalization import ActNorm
from nflows.transforms.conditional import ConditionalTransform
from nflows.utils import torchutils


class FillTriangular(Transform):

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.N = calc_numb_ltri_dim(features)
        self.lower_indices = np.tril_indices(self.N, k=0)

    def forward(self, inputs, context=None):
        mb = inputs.shape[0]
        outputs = inputs.new_zeros((mb, self.N, self.N))
        outputs[:, self.lower_indices[0], self.lower_indices[1]] = inputs

        if torch.any(torch.isnan(outputs)): breakpoint()

        logabsdet = inputs.new_zeros(mb)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        assert inputs.shape[1] == self.N

        outputs = inputs[:, self.lower_indices[0], self.lower_indices[1]]

        logabsdet = inputs.new_zeros(inputs.shape[0])

        return outputs, logabsdet

class PositiveDefiniteAndUnconstrained(Transform):

    def __init__(self, features, p_features, jitter=1e-7):
        super().__init__()
        self.jitter = jitter
        self.MAX_EXP = 88.
        self.features = features
        self.p_features = p_features
        self.P = calc_numb_ltri_dim(p_features)
        self.Q = (self.features - self.p_features) // self.P
        assert self.Q * self.P + self.p_features == self.features

        # indices for lower triangular entries
        self.lower_indices = np.tril_indices(self.P, k=0)

        # mask for sofplus over diagonal
        PxP_diag = torch.diag_embed(torch.ones(1, self.P))
        PxPplusQ_diag = F.pad(input=PxP_diag, pad=(0, self.Q, 0, 0), mode='constant', value=0)
        self.diag_mask_PxP = nn.Parameter(PxP_diag, requires_grad=False)
        self.diag_mask_PxPplusQ = nn.Parameter(PxPplusQ_diag, requires_grad=False)
        self.diag_indices = np.diag_indices(self.P)

        # powers for cholesky outer product jacobian determinant
        self.powers = nn.Parameter(torch.arange(self.P, 0, -1).unsqueeze(0), requires_grad=False)
        self.eye = nn.Parameter(torch.diag_embed(torch.ones(self.P)).unsqueeze(0), requires_grad=False)

    def forward(self, inputs, context=None):
        mb, features = inputs.shape
        assert features == self.features

        # 1) fill P x (P + Q) matrix
        outputs_fill = inputs.new_zeros((mb, self.P, self.P + self.Q))
        outputs_fill[:, self.lower_indices[0], self.lower_indices[1]] = inputs[:, :self.p_features]
        outputs_fill[:, :, self.P:] = inputs[:, self.p_features:].view(-1, self.P, self.Q)

        if torch.any(torch.isnan(outputs_fill)): breakpoint()

        # logabsdet is zero

        # 2) take softplus over diagonal of submatrix P x P
        softplus_diag = F.softplus(outputs_fill) + self.jitter
        outputs_soft = self.diag_mask_PxPplusQ * softplus_diag + (1. - self.diag_mask_PxPplusQ) * outputs_fill

        if torch.any(torch.isnan(outputs_soft)): breakpoint()

        logabsdet_soft = - (F.softplus(-torch.diagonal(outputs_fill, dim1=-2, dim2=-1)) + self.jitter).sum(-1)  # maybe jitter creates problems here

        # 3) compute Cholesky outer product on submatrix P x P
        self.check_pos_low_triang(outputs_soft)

        diagonal = torch.diagonal(outputs_soft, dim1=-2, dim2=-1)
        outputs = inputs.new_zeros((mb, self.P, self.P + self.Q))
        outputs[:, :self.P, :self.P] = outputs_soft[:, :self.P, :self.P] @ outputs_soft[:, :self.P, :self.P].mT
        outputs[:, :self.P, self.P:] = outputs_soft[:, :self.P, self.P:]

        if torch.any(torch.isnan(outputs)): breakpoint()

        logabsdet_chol = self.P * np.log(2.) + (self.powers * diagonal.log()).sum(-1)

        return outputs, logabsdet_soft + logabsdet_chol

    def inverse(self, inputs, context=None):
        assert inputs.shape[1] == self.P and inputs.shape[2] == self.P + self.Q

        # 1) compute Cholesky decomposition
        inputs_jitter = inputs[:,:self.P, :self.P] + self.eye * self.jitter
        self.check_pos_def(inputs_jitter)

        outputs_low_tri = torch.linalg.cholesky(inputs_jitter, upper=False)
        if torch.any(torch.isnan(outputs_low_tri)): breakpoint()

        diagonal = torch.diagonal(outputs_low_tri, dim1=-2, dim2=-1)
        logabsdet_chol = self.N * np.log(2.) + (self.powers * diagonal.log()).sum(1)

        # 2) inverse softplus over the diagonal
        self.check_pos_low_triang(outputs_low_tri)

        inv_softplus_diag = (outputs_low_tri.abs().clamp(max=self.MAX_EXP).exp() - 1. + self.jitter).log()
        outputs_inv_soft = self.diag_mask_PxP * inv_softplus_diag + (1. - self.diag_mask_PxP) * outputs_low_tri

        if torch.any(torch.isnan(outputs_inv_soft)): breakpoint()

        der_inv_softplus = 1. - torch.diagonal(- outputs_low_tri, dim1=-2, dim2=-1).clamp(max=self.MAX_EXP).exp() + self.jitter
        logabsdet_inv_soft = - der_inv_softplus.log().sum(-1)

        # 3) ravel P x (P + Q) matrix into a vector

        mb = inputs.shape[0]
        outputs = inputs.new_zeros((mb, self.p_features + self.P * self.Q))
        outputs[:, :self.p_features] = outputs_inv_soft[:, self.lower_indices[0], self.lower_indices[1]]
        outputs[:,self.p_features:] = inputs[:, :, self.P:]

        # logabsdet is zero

        return outputs, logabsdet_chol + logabsdet_inv_soft

    def check_pos_low_triang(self, inputs):
        upper_indices = np.triu_indices(self.P, k=1)
        assert torch.all(inputs[:, upper_indices[0], upper_indices[1]] == 0.), (
            "input tensor must be mini batch of lower triangular matrices")
        assert torch.all(torch.diagonal(inputs, dim1=-2, dim2=-1) > 0), (
            'input tensor must be mini batch of lower triangular matrices with positive diagonal elements')

    def check_pos_def(self, inputs):
        assert torch.all(inputs[:, :self.P, :self.P] == inputs[:, :self.P, :self.P].mT), (
            "input matrix must be symmetric"
        )
        assert  torch.all(torch.linalg.eig(inputs[:,:self.P, :self.P])[0].real >= 0), (
            "input matrix must be symmetric positive semi-definite in order to perform Cholesky decomposition"
        )


class FillSymmetricZeroDiag(Transform):

    def __init__(self, features):
        super().__init__()

        self.features = features
        self.N = calc_numb_ltri_dim(features) + 1 # zeros on diagonal
        self.lower_indices = np.tril_indices(self.N, k=-1)


    def forward(self, inputs, context=None):
        mb = inputs.shape[0]
        outputs = inputs.new_zeros((mb, self.N, self.N))
        outputs[:, self.lower_indices[0], self.lower_indices[1]] = inputs
        outputs = outputs + outputs.mT # symmetric matrix

        if torch.any(torch.isnan(outputs)): breakpoint()

        logabsdet = inputs.new_zeros(mb)

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        assert inputs.shape[1] == self.N

        outputs = inputs[:, self.lower_indices[0], self.lower_indices[1]]

        logabsdet = inputs.new_zeros(inputs.shape[0])

        return outputs, logabsdet


class SigmoidForSigma(Transform):
    """
    Transform that applies Sigmoid to last dimension. Works for 1D inputs
    [x_1, ..., x_n-1, x_n] --> [x_1, ..., x_n-1,  Sigmoid(x_n)]
    """
    def __init__(self, temperature=1, eps=1e-6, learn_temperature=False):
        super().__init__()
        self.eps = eps
        if learn_temperature:
            self.temperature = nn.Parameter(torch.Tensor([temperature]))
        else:
            temperature = torch.Tensor([temperature])
            self.register_buffer('temperature', temperature)

    def forward(self, inputs, context=None):
        #inputs = self.temperature * inputs
        output_non_sigma = inputs[...,:-1]
        output_sigma = torch.sigmoid(self.temperature * inputs[..., -1])
        outputs = torch.cat((output_non_sigma, output_sigma.unsqueeze(-1)), -1)
        # logabsdet = torchutils.sum_except_batch(
        #     torch.log(self.temperature) - F.softplus(-inputs[...,-1].unsqueeze(-1)) - F.softplus(inputs[...,-1].unsqueeze(-1))
        # )
        logabsdet = torch.log(self.temperature) - F.softplus(-inputs[...,-1]) - F.softplus(inputs[...,-1])

        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        if torch.min(inputs[...,-1]) < 0 or torch.max(inputs[...,-1]) > 1:
            raise InputOutsideDomain()

        inputs_sigma = torch.clamp(inputs[...,-1], self.eps, 1 - self.eps)

        output_non_sigma = inputs[..., :-1]
        output_sigma = (1 / self.temperature) * (torch.log(inputs_sigma) - torch.log1p(-inputs_sigma))
        outputs = torch.cat((output_non_sigma, output_sigma.unsqueeze(-1)), -1)

        # logabsdet = -torchutils.sum_except_batch(
        #     torch.log(self.temperature)
        #     - F.softplus(-self.temperature * output_sigma)
        #     - F.softplus(self.temperature * output_sigma)
        # )
        logabsdet = torch.log(self.temperature) - F.softplus(-self.temperature * output_sigma) \
                    - F.softplus(self.temperature * output_sigma)

        return outputs, logabsdet


class Softplus(Transform):
    def __init__(self, jitter=1e-7):
        super().__init__()
        self.jitter = jitter
        self.MAX_EXP = 88.

    def forward(self, inputs, context=None):
        #mask = torch.diag_embed(torch.ones(inputs.shape[:-1])).cuda() # (n_batches, d, d) --> (n_batches, d x d identity)
        #outputs = mask * (F.softplus(inputs) + self.jitter) + (1. - mask) * inputs

        inputs = F.softplus(inputs) + self.jitter

        if torch.any(torch.isnan(inputs)): breakpoint()

        logabsdet = - (F.softplus(-torch.diagonal(inputs)) + self.jitter).sum(-1) # maybe jitter creates problems here

        return inputs, logabsdet

    def inverse(self, inputs, context=None):

        assert torch.all(inputs + self.jitter > 0) , ('tensor to invert must have positive diagonal elements')

        inputs = (inputs.clamp(max=self.MAX_EXP).exp() - 1. + self.jitter).log()

        der_inv_softplus = 1. - (-inputs).clamp(max=self.MAX_EXP).exp() + self.jitter
        logabsdet = - der_inv_softplus.log().sum(-1)

        return inputs, logabsdet


class ActNormVector(Transform):
    def __init__(self, features, mu=None, std=None, raw_params: torch.Tensor=None):
        super().__init__()

        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("raw_parameters", torch.tensor(False, dtype=torch.bool))

        if raw_params is None:
            self.log_scale = nn.Parameter(torch.zeros(1, features), requires_grad=True)
            self.shift = nn.Parameter(torch.zeros(1, features), requires_grad=True)
        else:
            assert raw_params.shape[1:] == (features, 2)
            mb = raw_params.shape[0]
            self.log_scale_param = self.constrained_raw_params(raw_params[:,:,0])
            self.shift_param = self.constrained_raw_params(raw_params[:,:,1])
            self.log_scale = raw_params.new_zeros(mb, features)
            self.shift = raw_params.new_zeros(mb, features)

            with torch.no_grad():
                self.std = std
                self.mu = mu

            self.raw_parameters.data = torch.tensor(True, dtype=torch.bool)

    @property
    def scale(self):
        return torch.exp(self.log_scale)

    def constrained_raw_params(self, inputs):
        return 2 * (torch.sigmoid(inputs * 0.2) - 0.5) # inputs

    def compute_mu_std(self, inputs):
        with torch.no_grad():
            std = inputs.std(dim=0)
            mu = (inputs / std).mean(dim=0)
        return mu, std

    def forward(self, inputs, context=None):
        if self.training and not self.initialized:
            self._initialize(inputs)

        outputs = self.scale * inputs + self.shift

        if self.raw_parameters:
            logabsdet = self.log_scale.sum(1)
        else:
            logabsdet = self.log_scale.sum() * outputs.new_ones(inputs.shape[0])

        return outputs, logabsdet

    def inverse(self, inputs, context=None):

        outputs = (inputs - self.shift) / self.scale

        if self.raw_parameters:
            logabsdet = - self.log_scale.sum(1)
        else:
            logabsdet = - self.log_scale.sum() * outputs.new_ones(inputs.shape[0])

        return outputs, logabsdet

    def _initialize(self, inputs):
        """Data-dependent initialization, s.t. post-actnorm activations have zero mean and unit
        variance. """
        mu, std = self.compute_mu_std(inputs)
        if self.raw_parameters:
            if self.std is None and self.mu is None:
                self.log_scale = - torch.log(std) + self.log_scale_param
                self.shift = - mu + self.shift_param
            elif self.std is not None and self.mu is not None:
                self.log_scale = - torch.log(self.std) + self.log_scale_param
                self.shift = - self.mu + self.shift_param
            else:
                raise ValueError("Both std and mu must be either None or a specified tensor")
        else:
            with torch.no_grad():
                self.log_scale.data = - torch.log(std)
                self.shift.data = - mu
        with torch.no_grad():
            self.initialized.data = torch.tensor(True, dtype=torch.bool)


class ConditionalActNormVector(ConditionalTransform):
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
            transformer = ActNormVector(features=self.features, mu=self.mu, std=self.std,
                                        raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                              self._output_dim_multiplier()))
        else:
            transformer = ActNormVector(features=self.features,
                                        raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                              self._output_dim_multiplier()))
            self.mu, self.std = transformer.compute_mu_std(inputs)
            self.init_mu_std.data = torch.tensor(True, dtype=torch.bool)

        z, logabsdet = transformer(inputs)
        return z, logabsdet

    def _inverse_given_params(self, inputs, autoregressive_params):
        if self.init_mu_std:
            transformer = ActNormVector(features=self.features, mu=self.mu, std=self.std,
                                        raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                              self._output_dim_multiplier()))
        else:
            transformer = ActNormVector(features=self.features,
                                        raw_params=autoregressive_params.view(inputs.shape[0], self.features,
                                                                              self._output_dim_multiplier()))
            self.mu, self.std = transformer.compute_mu_std(inputs)
            self.init_mu_std.data = torch.tensor(True, dtype=torch.bool)

        x, logabsdet = transformer.inverse(inputs)
        return x, logabsdet


def inv_log_gamma (inputs: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor):
    """Inverse gamma distribution"""
    assert alpha > 0 and beta > 0, "alpha and beta coefficient must be positive"
    assert torch.all(inputs) >= 0, "input vector must be positive (check for non-negative)"

    eps = 1e-7
    log_norm = alpha * torch.log(beta) - torch.lgamma(alpha)
    log_unnorm = - (alpha + 1.) * torch.log(inputs) - beta / (inputs + eps)

    return log_norm + log_unnorm

def calc_numb_ltri_dim(p):
    assert p>0, "dimension must be positive number"
    temp = 1 + 8 * p
    assert np.square(np.floor(np.sqrt(temp))) == temp, "invalid dimension: can't be mapped to lower triangular matrix"
    N = int((-1 + np.floor(np.sqrt(temp))) // 2)

    return N

def block_indices(i, j, mb, p, q, cuda=True):
    assert i < p and j < p, "indices out of range: i,j  [0,1,...,p-1]"

    # indices for a 2-D matrix: [p*q, p*q]
    block_coord = torch.cartesian_prod(torch.arange(i*q, (i+1)*q), torch.arange(j*q, (j+1)*q)).long()
    row_indices = block_coord[:,0]
    col_indices = block_coord[:,1]

    # repeat indices for a batch of mb 2-D matrices: [mb, p*q, p*q]
    batch_indices = torch.stack([torch.ones(block_coord.shape[0]).long() * i for i in range(mb)]).ravel()
    row_indices = row_indices.repeat(mb)
    col_indices = col_indices.repeat(mb)

    if cuda:
        return batch_indices.cuda(), row_indices.cuda(), col_indices.cuda()
    else:
        return batch_indices, row_indices, col_indices

def test():
    n_dim = 4
    inputs = torch.randn(10, n_dim, n_dim)
    context = None # torch.randn(16, 24)
    transform = TransformDiagonalExponential()
    outputs, logabsdet = transform(inputs, context)

    # forward transforamtion
    func = lambda inputs: transform.forward(inputs)[0]
    inv_func = lambda inputs: transform.inverse(inputs)[0]

    logabsdet_greedy = [torch.autograd.functional.jacobian(func, inputs[i].unsqueeze(0)).squeeze().view(-1, n_dim * n_dim).det().abs().log()
                        for i in range(inputs.shape[0])]

    print(logabsdet)
    print(logabsdet_greedy)

    # other tests
    from nflows.transforms import FillTriangular, CholeskyOuterProduct
    import torch
    import numpy as np
    trans_triang = FillTriangular()
    trans_cholesky = CholeskyOuterProduct()
    func_cholesky = lambda inputs: trans_cholesky.forward(inputs)[0]
    func_cholesky_inv = lambda inputs: trans_cholesky.inverse(inputs)[0]
    A = torch.rand(4,3)
    A_ = trans_triang(A)[0]
    A__ = func_cholesky(A_)

    logabsdet = trans_cholesky.forward(A_)[1]
    logabsdet_inv = trans_cholesky.inverse(A__)[1]

    jac_cholesky = batch_matrix_jacobian(func_cholesky, A_)
    jac_cholesky_inv = batch_matrix_jacobian(func_cholesky_inv, A__)


if __name__ == "__main__":
    test()


# deprecated - use only with jacobian wrt vectors, not matrices
def batch_matrix_jacobian(func, input):
    assert len(input.shape) == 3, "input tensor must be mini batch of matrices: (mb, n, m)"
    jac = []
    for d in range(input.shape[0]):
        out_dim = np.prod(input.shape[1:])
        jac.append(torch.autograd.functional.jacobian(func, input[d].unsqueeze(0)).squeeze().view(out_dim, out_dim).unsqueeze(0))
    return torch.cat(jac)