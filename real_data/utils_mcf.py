import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import tqdm
import pickle
import pyreadr
import pandas as pd
from gglasso.solver.single_admm_solver import ADMM_SGL

import utils_plot

from datetime import timedelta
from enflows.distributions import StandardNormal, Uniform, ConditionalDiagonalNormal
from enflows.transforms import FillTriangular, RandomPermutation, MaskedSumOfSigmoidsTransform, LipschitzDenseNetBuilder, iResBlock
from enflows.transforms.normalization import ActNorm
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.matrix import TransformDiagonalSoftplus, CholeskyOuterProduct, PositiveDefiniteAndUnconstrained
from enflows.flows.base import Flow
from enflows.nn.nets import ResidualNet, Sin
from sklearn.linear_model import LinearRegression, Lasso, Ridge


def calc_num_lower_tri_entries(matrix_dim, include_diag=True):
    N = matrix_dim
    if include_diag:
        return N * (N + 1) // 2
    else:
        return N * (N - 1) // 2

def gen_cooling_schedule(T0, Tn, num_iter, scheme):
    def cooling_schedule(t):
        if t < num_iter:
            k = t / num_iter
            if scheme == 'exp_mult':
                alpha = Tn / T0
                return T0 * (alpha ** k)
            #elif scheme == 'log_mult':
            #    return T0 / (1 + alpha * math.log(1 + k))
            elif scheme == 'lin_mult':
                alpha = (T0 / Tn - 1)
                return T0 / (1 + alpha * k)
            elif scheme == 'quad_mult':
                alpha = (T0 / Tn - 1)
                return T0 / (1 + alpha * (k ** 2))
        else:
            return Tn
    return cooling_schedule

def build_positive_definite_vector (matrix_dim, n_layers=3, context_features=16, hidden_features=256, device='cuda'):
    # base distribution over flattened triangular matrix
    flow_dim = calc_num_lower_tri_entries(matrix_dim)
    base_dist = StandardNormal(shape=[flow_dim])

    # Define an invertible transformation
    transformation_layers = []

    for _ in range(n_layers):
        transformation_layers.append(
            InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim, hidden_features=hidden_features,
                                                  context_features=context_features, num_blocks=5, n_sigmoids=30)
            )
        )
        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim)
            )
        )


    transformation_to_cholesky = [
        InverseTransform(FillTriangular(features=flow_dim)),
        InverseTransform(TransformDiagonalSoftplus(N=matrix_dim)),
        InverseTransform(CholeskyOuterProduct(N=matrix_dim)),
    ]

    transformation_layers = transformation_layers + transformation_to_cholesky
    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNet(in_features=2, out_features=context_features, hidden_features=64,
                                num_blocks=5, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(device)

    return flow


def build_cond_psd_uncontrained_vector (P, Q, n_layers=3, context_features=16, hidden_features=256, device='cuda'):
    # base distribution over flattened triangular matrix
    P_low_tri_entries = calc_num_lower_tri_entries(P)
    flow_dim = P_low_tri_entries + P * Q
    base_dist = StandardNormal(shape=[flow_dim])

    # Define an invertible transformation
    transformation_layers = []

    transformation_layers.append(
        InverseTransform(
            ActNorm(features=flow_dim)
        )
    )

    for _ in range(n_layers):
        transformation_layers.append(
            RandomPermutation(flow_dim)
        )
        transformation_layers.append(
            InverseTransform(
                MaskedSumOfSigmoidsTransform(features=flow_dim, hidden_features=hidden_features,
                                                  context_features=context_features, num_blocks=3, n_sigmoids=30)
            )
        )
        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim)
            )
        )

    transformation_to_cholesky = [
        InverseTransform(PositiveDefiniteAndUnconstrained(flow_dim, P_low_tri_entries))
    ]

    transformation_layers = transformation_layers + transformation_to_cholesky
    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNet(in_features=2, out_features=context_features, hidden_features=64,
                                num_blocks=5, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(device)

    return flow


def build_positive_definite_vector_lipschitz (matrix_dim, n_layers=3, context_features=16, hidden_features=256, device='cuda'):
    # base distribution over flattened triangular matrix
    flow_dim = calc_num_lower_tri_entries(matrix_dim)
    base_dist = StandardNormal(shape=[flow_dim])

    densenet_builder = LipschitzDenseNetBuilder(input_channels=flow_dim,
                                                densenet_depth=3,
                                                densenet_growth=flow_dim+10,
                                                activation_function=Sin(w0=20),
                                                lip_coeff=.97,
                                                context_features=context_features
                                                )


    class TimeNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tanh = torch.nn.Sigmoid()
            self.scaler = torch.nn.Linear(context_features, 1)

        def forward(self, inputs):
            return self.tanh(self.scaler((inputs - 0.5) * 2))

    # Define an invertible transformation
    transformation_layers = []

    for _ in range(n_layers):
        transformation_layers.append(
            InverseTransform(
                iResBlock(densenet_builder.build_network(), time_nnet=TimeNetwork(), brute_force=False)
            )
        )
        transformation_layers.append(
            InverseTransform(
                ActNorm(features=flow_dim)
            )
        )


    transformation_to_cholesky = [
        InverseTransform(FillTriangular(features=flow_dim)),
        InverseTransform(TransformDiagonalSoftplus(N=matrix_dim)),
        InverseTransform(CholeskyOuterProduct(N=matrix_dim)),
    ]

    transformation_layers = transformation_layers + transformation_to_cholesky
    transformation_layers = transformation_layers[::-1]
    transform = CompositeTransform(transformation_layers)

    # define embedding (conditional) network
    embedding_net = ResidualNet(in_features=2, out_features=context_features, hidden_features=64,
                                num_blocks=5, activation=torch.nn.functional.relu)

    # combine into a flow
    flow = Flow(transform, base_dist, embedding_net=embedding_net).to(device)

    return flow

def log_likelihood_unnormalized(W: torch.Tensor, S: torch.Tensor, n):
    """
    \propto p(S|W)
    :param W: precision matrix W
    :param S: Sample Covariance Matrix
    :return: unnormalized log likelihood
    """

    jitter = 1e-3
    eye_like_W = torch.diag_embed(W.new_ones(*W.shape[:-1]))
    W_jitter = W + eye_like_W * jitter

    try:
        L = torch.linalg.cholesky(W_jitter)
    except:
        breakpoint()

    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
    log_det_W = 2 * diag_L.log().sum(-1)
    tr_SW = (S @ W).diagonal(dim1=-2, dim2=-1).sum(-1)

    return (log_det_W - tr_SW) * n * 0.5

def log_prior_generalized_normal(W: torch.Tensor, lamb_exp: torch.Tensor, p: torch.Tensor, diagonal=True) -> torch.Tensor:
    """
    \propto p(W|λ)
    :param W: precision matrix W
    :param lamb_exp: lambda parameter (lambda = 10 ** lamb_exp)
    :return: prior over W - off-diag elements follow generalized normal (and also diagonal if diagonal=True)
    """
    if diagonal: # regularizes diagonal elements as well
        tril_indices = np.tril_indices(W.shape[-1], k=0)
        n_elements = 0.5 * W.shape[-1] * (W.shape[-1] + 1)
    else:
        tril_indices = np.tril_indices(W.shape[-1], k=-1)
        n_elements = 0.5 * W.shape[-1] * (W.shape[-1] - 1)

    W_tril = W[..., tril_indices[0], tril_indices[1]]

    # compute prior
    eps = 1e-7
    lamb = 10 ** lamb_exp

    p_reshaped_W = p.view(-1, *len(W_tril.shape[1:]) * (1,))  # (-1, 1, 1)
    W_tril_p = (W_tril.abs() + eps).pow(p_reshaped_W).sum(-1)

    W_diag = torch.diagonal(W, dim1=-2, dim2=-1) # diagonal elements with 0.5 lambda regularization
    W_diag_p = (W_diag.abs()).sum(-1)  # .pow(p_reshaped_W).sum(-1)

    log_prior_gen_norm = - lamb * W_tril_p - 0.5 * lamb * W_diag_p

    # compute normalization constant
    if torch.any(torch.isnan(log_prior_gen_norm)): breakpoint()
    norm_const = (0.5 * p).log() + lamb.log() / p - torch.lgamma(1. / p)
    if torch.any(torch.isnan(norm_const)): breakpoint()

    if diagonal:
        norm_const_diag = (0.5 * lamb).log()
        log_const = n_elements * norm_const + W.shape[-1] * norm_const_diag
    else:
        log_const = n_elements * norm_const

    return log_prior_gen_norm + log_const


def log_prior_generalized_normal_(W: torch.Tensor, lamb_exp: torch.Tensor, p: torch.Tensor, triangular=True, diagonal=False) -> torch.Tensor:
    """
    \propto p(W|λ)
    :param W: precision matrix W
    :param lamb_exp: lambda parameter (lambda = 10 ** lamb_exp)
    :return: prior over W - off-diag elements follow generalized normal (and also diagonal if diagonal=True)
    """
    if triangular:
        if diagonal: # regularizes diagonal elements as well
            tril_indices = np.tril_indices(W.shape[-1], k=0)
            n_elements = 0.5 * W.shape[-1] * (W.shape[-1] + 1)
        else:
            tril_indices = np.tril_indices(W.shape[-1], k=-1)
            n_elements = 0.5 * W.shape[-1] * (W.shape[-1] - 1)

        W_tril = W[..., tril_indices[0], tril_indices[1]]
    else:
        n_elements = np.prod(W.shape[-2:])
        W_tril = W.reshape(*W.shape[:-2], n_elements)

    # compute prior
    eps = 1e-7
    lamb = 10 ** lamb_exp
    p_reshaped_W = p.view(-1, *len(W_tril.shape[1:]) * (1,))  # (-1, 1, 1)
    W_tril_p = (W_tril.abs() + eps).pow(p_reshaped_W).sum(-1)
    log_prior_gen_norm = - lamb * W_tril_p

    # compute normalization constant
    if torch.any(torch.isnan(log_prior_gen_norm)): breakpoint()
    norm_const = (0.5 * p).log() + lamb.log() / p - torch.lgamma(1. / p)
    if torch.any(torch.isnan(norm_const)): breakpoint()
    log_const = n_elements * norm_const

    return log_prior_gen_norm + log_const


def log_prior_exp_diag(W: torch.Tensor, lamb_exp: torch.Tensor, p: torch.Tensor, diagonal=True) -> torch.Tensor:
    """
    \propto p(W|λ,p)
    :param W: precision matrix W
    :param lamb_exp: lambda parameter (lambda = 10 ** lamb_exp)
    :return: prior over W - diag elements follow laplacian
    """
    diag = torch.diagonal(W, dim1=-2, dim2=-1)
    lamb = 10 ** lamb_exp
    log_prior_exp = - 0.5 * lamb * diag.sum(-1)

    log_const = 0.5 * W.shape[-1] * torch.log(0.5 * lamb)

    return log_prior_exp + log_const

def log_unnorm_posterior(W, S, lamb_exp, p, n):
    """
    \propto p(W|S,λ,p)
    :param W: precision matrix W
    :param S: Sample Covariance Matrix
    :param lamb: lambda parameter (lambda = 10 ** lamb_exp)
    :return: unnormalized posterior over W, given by product (sum of log) of likelihood and prior
    """
    likelihood = log_likelihood_unnormalized(W=W, S=S, n=n)
    # diag_prior = log_prior_exp_diag(W=W, lamb_exp=lamb_exp, p=p)
    # prior = log_prior_wang(W=W, T=T, lamb=lamb)
    off_diag_prior = log_prior_generalized_normal(W=W, lamb_exp=lamb_exp, p=p, diagonal=False)

    return likelihood + off_diag_prior # + diag_prior


def log_unnorm_posterior_p(W: torch.Tensor, S: torch.Tensor, P, Q, n, lamb_exp, p):
    """
    \propto p(W_11, W_12 | S, lambda)
    :param W: precision matrix W
    :param S: Sample Covariance Matrix
    :return: unnormalized log likelihood
    """
    assert W.shape[-2] == P and W.shape[-1] == P + Q

    W11 = W[..., :P, :P]
    W12 = W[..., :P, P:]

    Sn = S * n
    S11 = Sn[:P, :P].unsqueeze(0)
    S12 = Sn[:P, P:].unsqueeze(0)
    S22 = Sn[P:, P:].unsqueeze(0)

    eye_like_W11 = torch.diag_embed(W11.new_ones(*W11.shape[:-1]))
    eye_like_S11 = torch.diag_embed(S11.new_ones(*S11.shape[:-1]))
    eye_like_S22 = torch.diag_embed(S22.new_ones(*S22.shape[:-1]))

    # log det W11
    jitter = 1e-3
    W11_jitter = W11 + eye_like_W11 * jitter
    L = torch.linalg.cholesky(W11_jitter)
    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
    log_det_W11 = 2 * diag_L.log().sum(-1)

    # -0.5 Tr (W11(S11 + 1) + 2 (W12^T @ S12) + W12^T @ W11^-1 @ W12 (S22 + 1))
    T1 = W11 @ (S11 + eye_like_S11)
    T2 = 2 * (W12.mT @ S12)
    T3 = W12.mT @ torch.linalg.solve(W11_jitter, W12 @ (S22 + eye_like_S22)) # numerically stable version

    trace_T1 = T1.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_T2 = T2.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace_T3 = T3.diagonal(dim1=-2, dim2=-1).sum(-1)
    trace = trace_T1 + trace_T2 + trace_T3

    # prior W11
    log_prior_W11 = log_prior_generalized_normal_(W11, lamb_exp, p, triangular=True, diagonal=False)

    # prior W12
    log_prior_W12 = log_prior_generalized_normal_(W12, lamb_exp, p, triangular=False)

    # log_prior_const = (0.5 * P * (P - 1) + P * Q ) * torch.log(0.5 * lamb)

    return 0.5 * n * log_det_W11 - 0.5 * trace + log_prior_W11 + log_prior_W12 #+ log_prior_const

def save_model (model, file_name=None, folder_name=None):
    if folder_name is None:
        folder_name = "./models/"

    if file_name is None:
        file_name = ''

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    torch.save(model.state_dict(), f"{folder_name}cmf_{file_name}")

def compute_glasso_solution(S, alpha_sorted, folder_name=None):

    if folder_name is None:
        folder_name = "./models/"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    file_name = "glasso_solution.pkl"
    if os.path.exists(folder_name+file_name):
        with open(folder_name+file_name, 'rb') as f:
            sol_dict = pickle.load(f)
    else:
        print("Computing the frequentist solution (GLasso) for the full matrix. This might take a while...")
        print("Once computed it is saved and just loaded in following runs")
        glasso_solution = np.array([ADMM_SGL(S.detach().cpu().numpy(), lamb * 0.5, np.eye(S.shape[0]))[0]['Theta'] for lamb in tqdm.tqdm(alpha_sorted)])
        sol_dict = dict(W=glasso_solution, alphas=alpha_sorted)
        with open(folder_name+file_name, 'wb') as fp:
            pickle.dump(sol_dict, fp)

    return sol_dict




def train_model(model, S, P, Q, n, epochs=2_001, T0=5., Tn=.001, iter_per_cool_step=100, lr=1e-3, sample_size=1, context_size=1_000,
                p_min=0.01, p_max=3., lambda_min_exp=-2, lambda_max_exp=1, device="cuda", file_name=None, **kwargs):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # set up cooling schedule
    num_iter = epochs // iter_per_cool_step
    cooling_function = gen_cooling_schedule(T0=T0, Tn=Tn, num_iter=num_iter - 1, scheme='exp_mult')

    loss, loss_T = [], []
    try:
        start_time = time.monotonic()
        for epoch in range(epochs):
            T = cooling_function(epoch // (epochs / num_iter))

            optimizer.zero_grad()
            uniform_lambdas = torch.rand(context_size).to(device)
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            uniform_p = torch.rand(context_size) * (p_max - p_min) + p_min
            uniform_p = uniform_p.to(device).view(-1, 1)
            context = torch.cat((lambdas_exp, uniform_p), 1)

            q_samples_W, q_log_prob_W = model.sample_and_log_prob(num_samples=sample_size, context=context)
            posterior_eval = log_unnorm_posterior_p(q_samples_W, S=S, P=P, Q=Q, n=n, lamb_exp=lambdas_exp, p=uniform_p)

            kl_div = torch.mean(q_log_prob_W - posterior_eval / T)

            kl_div.backward()
            optimizer.step()

            loss_T.append(kl_div.cpu().detach().numpy())
            loss.append(torch.mean(q_log_prob_W - posterior_eval).cpu().detach().numpy())

            if epoch % (epochs//25) == 0:
                print(scheduler.get_last_lr())
                scheduler.step()

            if epoch % 100 == 0:
                print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")
                print(f"q_log_prob_W: {q_log_prob_W.mean().cpu().detach().numpy():.1f} "
                      f"posterior_eval: {posterior_eval.mean().cpu().detach().numpy():.1f}")

            T_1 = 1
            T_1_condition = T >= T_1 and cooling_function((epoch + 1) // (epochs / num_iter)) < T_1
            if epoch == epochs-1 or T_1_condition:
                save_model (model, file_name=file_name)
                p = uniform_p.new_ones(1)
                utils_plot.plot_full_comparison(model, S=S, P=P, Q=Q, n=n, T=T, p=p, lambda_min=lambda_min_exp,
                                                lambda_max=lambda_max_exp, file_name=file_name, plot_mll=True)


    except KeyboardInterrupt:
        print("interrupted..")

    save_model (model, file_name=file_name)

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, loss, loss_T
def sample_W_fixed_p (model, S, P, Q, n, T, p, context_size=10, sample_size=100, n_iterations=500, lambda_min_exp=-2, lambda_max_exp=1):
    # Sample from approximate posterior & estimate significant edges via  posterior credible interval
    samples, lambda_list, kl_list, kl_T_list = [], [], [], []

    with torch.no_grad():
        for _ in tqdm.tqdm(range(n_iterations)):
            uniform_lambdas = torch.rand(context_size).cuda()
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            uniform_p = lambdas_exp.new_ones(context_size).view(-1, 1) * p
            context = torch.cat((lambdas_exp, uniform_p), 1)
            posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size, context=context)
            posterior_eval = log_unnorm_posterior_p(posterior_samples, S=S, P=P, Q=Q, n=n, lamb_exp=lambdas_exp, p=uniform_p)
            kl_div = log_probs_samples - posterior_eval
            kl_div_T = log_probs_samples - posterior_eval / T
            samples.append(posterior_samples.cpu().detach().numpy())
            # lambda_list.append((lambdas**p).view(-1).cpu().detach().numpy())
            lambda_list.append((10**lambdas_exp).view(-1).cpu().detach().numpy())
            kl_list.append(kl_div.cpu().detach().numpy())
            kl_T_list.append(kl_div_T.cpu().detach().numpy())

    # samples from posterior
    samples, lambda_list = np.concatenate(samples, 0), np.concatenate(lambda_list, 0)
    W_mean, W_std = samples.mean(1), samples.std(1)

    # kl for marginal likelihood
    kl_list, kl_T_list = np.concatenate(kl_list, 0), np.concatenate(kl_T_list, 0)

    lambda_sorted_idx = lambda_list.argsort()
    lambda_sorted = lambda_list[lambda_sorted_idx]
    W_mean, W_std = W_mean[lambda_sorted_idx], W_std[lambda_sorted_idx]
    kl, kl_T = kl_list[lambda_sorted_idx], kl_T_list[lambda_sorted_idx]
    samples_sorted = samples[lambda_sorted_idx]

    return samples_sorted, kl, kl_T, W_mean, W_std, lambda_sorted


def sample_W_fixed_lamb_and_p (model, S, p, exp_lamb, sample_size=100, n_iterations=500):
    # Sample from approximate posterior & estimate significant edges via  posterior credible interval
    samples = []

    with torch.no_grad():
        for _ in tqdm.tqdm(range(n_iterations)):
            exp_lambda = S.new_ones(1,1) * exp_lamb
            uniform_p = S.new_ones(1,1) * p
            context = torch.cat((exp_lambda, uniform_p), 1)
            posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size, context=context)
            samples.append(posterior_samples[0].cpu().detach().numpy())

    # samples from posterior
    samples = np.concatenate(samples, 0)

    return samples