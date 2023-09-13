import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import os
import tqdm

import utils_plot

from datetime import timedelta
from enflows.distributions import StandardNormal, Uniform
from enflows.transforms import FillTriangular, RandomPermutation, MaskedSumOfSigmoidsTransform
from enflows.transforms.normalization import ActNorm
from enflows.transforms.base import CompositeTransform, InverseTransform
from enflows.transforms.matrix import TransformDiagonalSoftplus, CholeskyOuterProduct
from enflows.flows.base import Flow
from enflows.nn.nets import ResidualNet
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

def save_model (model, file_name=None, folder_name=None):
    if folder_name is None:
        folder_name = "./models/"

    if file_name is None:
        file_name = ''

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    torch.save(model.state_dict(), f"{folder_name}cmf_{file_name}")

def save_alpha (alpha, file_name=None, folder_name=None):
    if folder_name is None:
        folder_name = "./models/"

    if file_name is None:
        file_name = ''

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    f = open(f"{folder_name}cmf_{file_name}.txt", "a")
    f.write(f"{alpha}\n")
    f.close()

def optimal_alpha(X, alpha_sorted, kl_T_mean):

    return alpha_sorted[np.argmin(kl_T_mean)]

def train_model(model, S, X, d, n, epochs=2_001, T0=5., Tn=.001, iter_per_cool_step=100, lr=1e-3, sample_size=1, context_size=1_000,
                p_min=0.01, p_max=3., lambda_min_exp=-2, lambda_max_exp=1, device="cuda", seed=1234):
    file_name = f'd{d}_n{n}_e{epochs}_pmin{p_min}_pmax{p_max}_lmin{lambda_min_exp}_lmax{lambda_max_exp}_seed{seed}'
    n = X.shape[0]
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
            posterior_eval = log_unnorm_posterior(q_samples_W, S=S, lamb_exp=lambdas_exp, p=uniform_p, n=n)

            kl_div = torch.mean(q_log_prob_W - posterior_eval / T)

            kl_div.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            loss_T.append(kl_div.cpu().detach().numpy())
            loss.append(torch.mean(q_log_prob_W - posterior_eval).cpu().detach().numpy())

            if epoch % (epochs//25) == 0 :
                print(scheduler.get_last_lr())
                scheduler.step()

            if epoch % 100 == 0:
                print(f"Training loss at step {epoch}: {loss[-1]:.1f} and {loss_T[-1]:.1f} * (T = {T:.3f})")
                print(f"q_log_prob_W: {q_log_prob_W.mean().cpu().detach().numpy():.1f} "
                      f"posterior_eval: {posterior_eval.mean().cpu().detach().numpy():.1f}")

            T_1 = 1
            T_1_condition = T >= T_1 and cooling_function((epoch+1) // (epochs / num_iter)) < T_1

            if T_1_condition:
                save_model (model, file_name=file_name+f'_T{T:.3f}')
                p = uniform_p.new_ones(1)
                n = X.shape[0]
                samples, kl_mean, kl_T_mean, kl_std, kl_T_std, W_mean, W_std, lambda_sorted = sample_W_fixed_p(model, S, T=T, p=p, n=n,
                                                                                                       context_size=2,
                                                                                                       sample_size=50,
                                                                                                       lambda_min_exp=lambda_min_exp,
                                                                                                       lambda_max_exp=lambda_max_exp)
                alpha_sorted = lambda_sorted * 2 / n
                utils_plot.plot_W_fixed_p(model, S, p=p, T=T, lamb_min=lambda_min_exp, lamb_max=lambda_max_exp, X_train=X,
                               n_plots=5)
                alpha_scikit = utils_plot.plot_log_likelihood(X, alpha_sorted, kl_T_mean, kl_T_std)

            if epoch == epochs-1:
                p = uniform_p.new_ones(1)
                n = X.shape[0]
                samples, kl_mean, kl_T_mean, kl_std, kl_T_std, W_mean, W_std, lambda_sorted = sample_W_fixed_p(model, S, T=T, p=p, n=n, context_size=2,
                                                                                                               sample_size=50,
                                                                                                               lambda_min_exp=lambda_min_exp,
                                                                                                               lambda_max_exp=lambda_max_exp)
                alpha_sorted = lambda_sorted * 2 / n
                opt_alpha = optimal_alpha(X, alpha_sorted, kl_T_mean)
                save_alpha(opt_alpha, file_name=file_name)


    except KeyboardInterrupt:
        print("interrupted..")

    save_model (model, file_name=file_name+f'_T{T:.3f}')

    end_time = time.monotonic()
    time_diff = timedelta(seconds=end_time - start_time)
    print(f"Training took {time_diff} seconds")

    return model, loss, loss_T

def sample_W_fixed_p (model, S, T, p, n, context_size=10, sample_size=100, lambda_min_exp=-2, lambda_max_exp=1):
    # Sample from approximate posterior & estimate significant edges via  posterior credible interval
    samples, lambda_list, kl_list, kl_T_list = [], [], [], []

    with torch.no_grad():
        for _ in tqdm.tqdm(range(200)):
            uniform_lambdas = torch.rand(context_size).cuda()
            lambdas_exp = (uniform_lambdas * (lambda_max_exp - lambda_min_exp) + lambda_min_exp).view(-1, 1)
            uniform_p = lambdas_exp.new_ones(context_size).view(-1, 1) * p
            context = torch.cat((lambdas_exp, uniform_p), 1)
            posterior_samples, log_probs_samples = model.sample_and_log_prob(sample_size, context=context)
            posterior_eval = log_unnorm_posterior(posterior_samples, S=S, lamb_exp=lambdas_exp, p=uniform_p, n=n)
            kl_div = log_probs_samples - posterior_eval
            kl_div_T = log_probs_samples - posterior_eval / T
            samples.append(posterior_samples.cpu().detach().numpy())
            # lambda_list.append((10**(lambdas_exp*p)).view(-1).cpu().detach().numpy())
            lambda_list.append((10**lambdas_exp).view(-1).cpu().detach().numpy())
            kl_list.append(kl_div.cpu().detach().numpy())
            kl_T_list.append(kl_div_T.cpu().detach().numpy())

    # samples from posterior
    samples, lambda_list = np.concatenate(samples, 0), np.concatenate(lambda_list, 0)
    W_mean, W_std = samples.mean(1), samples.std(1)

    # kl for marginal likelihood
    kl_list, kl_T_list = np.concatenate(kl_list, 0), np.concatenate(kl_T_list, 0)
    kl_mean, kl_T_mean = kl_list.mean(1), kl_T_list.mean(1)
    kl_std, kl_T_std = kl_list.std(1), kl_T_list.std(1)

    lambda_sorted_idx = lambda_list.argsort()
    lambda_sorted = lambda_list[lambda_sorted_idx]
    W_mean, W_std = W_mean[lambda_sorted_idx], W_std[lambda_sorted_idx]
    kl_mean, kl_T_mean = kl_mean[lambda_sorted_idx], kl_T_mean[lambda_sorted_idx]
    kl_std, kl_T_std = kl_std[lambda_sorted_idx], kl_T_std[lambda_sorted_idx]

    return samples, kl_mean, kl_T_mean, kl_std, kl_T_std, W_mean, W_std, lambda_sorted
