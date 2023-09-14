import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import os
import torch

import utils_mcf

import tqdm
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from gglasso.solver.single_admm_solver import ADMM_SGL
from sklearn.covariance import graphical_lasso, GraphicalLassoCV

r = robjects.r
r['source']('BayesLassoMCMC.R')
glasso_path = robjects.globalenv['GLassoPath']
bayes_lasso = robjects.globalenv['Bayes.glasso.MB']

def plot_adjacency_matrix(edge_weights, ax=None) -> None:

    graph = nx.from_numpy_array(edge_weights)

    pos = nx.circular_layout(graph)

    colors = {-1: "red",
              1: "lime"}

    weights = nx.get_edge_attributes(graph,'weight').values()
    nx.draw(graph, pos, edge_color=[colors[np.sign(w)] for w in weights], ax=ax)
    if ax is None:
      plt.axis('equal')
    else:
      ax.set_aspect('equal')

def plot_W_gt (W):
    sns.set_style("white")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    im = axs[0].imshow(np.sign(W), cmap=mpl.colors.ListedColormap(['white', "gray", 'black']))
    axs[0].set_ylabel("Sign of Matrix Entry")
    fig.colorbar(im, ax=axs[0], ticks=[-1, 0, 1])
    axs[0].set_title("Precision $W$")

    plot_adjacency_matrix(np.sign(W) - np.eye(W.shape[0]), ax=axs[1])
    axs[1].set_title("Dependency Graph $W$")

    plt.savefig('precision_ground_truth.pdf')
    plt.clf()
    # plt.show()

def glasso_solution (S, W, alpha=1e-2, mode='cd', plots=True, scikit=False):
    """# Graphical Lasso: Penalized Maximum Likelihood estimate"""
    if scikit:
        cov_gl, prec_gl = graphical_lasso(emp_cov=S, alpha=alpha, mode=mode)
    else:
        prec_gl = ADMM_SGL(S, alpha, np.eye(S.shape[0]))[0]['Theta']

    if plots:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_adjacency_matrix(np.sign(prec_gl) - np.eye(W.shape[0]), ax=axs[0])
        axs[0].set_title("Graphical Lasso Estimate")

        plot_adjacency_matrix(np.sign(W) - np.eye(W.shape[0]), ax=axs[1])
        axs[1].set_title("Ground truth")

        plt.savefig('precision_glasso_reconstruction.pdf')
        plt.clf()
        # plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        plot_adjacency_matrix(np.sign(prec_gl) - np.eye(W.shape[0]), ax=axs[0])
        axs[0].set_title("Graphical Lasso Estimate")

        plot_adjacency_matrix((np.sign(W) != np.sign(prec_gl)) * 1., ax=axs[1])
        axs[1].set_title("Connections that are wrongly estimated")

        plt.savefig('precision_glasso_error.pdf')
        plt.clf()
        # plt.show()

    return prec_gl

def plot_loss (loss, loss_T):
    N = len(loss)
    plt.plot(np.linspace(1, N, N), loss, label='loss/T')
    plt.plot(np.linspace(1, N, N), loss_T, label='loss')
    plt.yscale("log")
    plt.title("Loss per epoch")
    plt.savefig('loss_epoch.pdf')
    plt.clf()
    # plt.show()

def plot_log_likelihood(X, alpha_sorted, kl_T_mean, kl_T_std, folder_name=None):

    if folder_name is None:
        folder_name = "./plots/"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print("Computing best lambda with 5-fold cross-validation GLasso...")
    alpha_scikit = GraphicalLassoCV(alphas=alpha_sorted * 0.5).fit(X).alpha_

    fig, ax = plt.subplots()
    ax.plot(alpha_sorted, -kl_T_mean, alpha=0.7)
    kl_min, kl_max = (-kl_T_mean - 2*kl_T_std).min(), (-kl_T_mean + 2*kl_T_std).max()
    ax.fill_between(alpha_sorted, -kl_T_mean - 2*kl_T_std, -kl_T_mean + 2*kl_T_std, facecolor='b', alpha=0.3)
    plt.xscale('log')
    plt.xlabel(r'$\lambda$', fontsize=18)
    plt.ylabel(r'$\log p(X|\lambda)$', fontsize=18)
    plt.locator_params(axis='y', nbins=4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.vlines(alpha_scikit, kl_min, kl_max, label='GLasso CV', colors='r')
    plt.vlines(alpha_sorted[np.argmin(kl_T_mean)], kl_min, kl_max, label='flow estimate', colors='b')
    plt.legend()
    plt.savefig(f"{folder_name}W_marginal_likelihood.pdf", bbox_inches='tight')
    plt.clf()
    # plt.show()
    print(f"optimal lambdas: {alpha_sorted[np.argmin(kl_T_mean)]} {alpha_scikit}")

def plot_W_comparison_old(W_mean, W_std, W_sklearn, lambdas, p, T, off_diagonal=True, folder_name=None, n_plots=3):

    if folder_name is None:
        folder_name = "./plots/"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    n_samples = W_mean.shape[0]
    N = W_mean.shape[1]
    if off_diagonal:
        indices = np.tril_indices(N, k=-1)
    else:
        indices = np.diag_indices(N)
    tril_indices = (..., indices[0], indices[1])
    W_tril_mean = W_mean[tril_indices].reshape(n_samples, -1)
    W_tril_std = W_std[tril_indices].reshape(n_samples, -1)
    W_tril_sklearn = W_sklearn[tril_indices].reshape(n_samples, -1)

    MSE_lambda = np.sqrt(np.mean(np.square(W_tril_mean - W_tril_sklearn), 1))
    # print("MSE_lambda for each lambda", MSE_lambda)
    print("MSE_lambda mean", MSE_lambda.mean())

    # plot precision matrix entries as a function of lambda
    j = 0
    n_lines = W_tril_mean.shape[1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots()
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == W_tril_mean.shape[1]:
                    break
                mean = W_tril_mean[:, j]
                std = W_tril_std[:, j]
                color = clrs[j % n_lines]
                ax.plot(lambdas, mean, c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(lambdas, mean - 2*std, mean + 2*std, alpha=0.2, facecolor=color)
                ax.plot(lambdas, W_tril_sklearn[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            ax.set_xscale('log')
            plt.xlabel(r'$\lambda$', fontsize=18)
            plt.ylabel(r'$\Omega$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(f"{folder_name}W_lambda_p{p.item():.2f}_T{T:.3f}_{i}.pdf", bbox_inches='tight')
            plt.close()

    # # compute norms
    # W_tril_norm = np.power(np.power(np.abs(W_tril_mean), p).sum(1), 1./p)
    # W_tril_norm /= W_tril_norm.max()
    # W_tril_norm_sorted_idx = W_tril_norm.argsort()
    # W_tril_norm = W_tril_norm[W_tril_norm_sorted_idx]
    # W_tril_norm_mean = W_tril_mean[W_tril_norm_sorted_idx]
    # W_tril_norm_std = W_tril_std[W_tril_norm_sorted_idx]
    #
    # sklearn_norm = np.power(np.power(np.abs(W_tril_sklearn), p).sum(1), 1./p)
    # sklearn_norm /= sklearn_norm.max()
    # sklearn_sorted_idx = sklearn_norm.argsort()
    # sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    # sklearn_sorted = W_tril_sklearn[sklearn_sorted_idx]
    #
    # for i in range(n_plots):
    #     fig, ax = plt.subplots()
    #     with sns.axes_style("darkgrid"):
    #         for j in range(i * n_lines, (i + 1) * n_lines):
    #             if j == W_tril_mean.shape[1]:
    #                 break
    #             std, mean = W_tril_norm_std[:, j], W_tril_norm_mean[:, j]
    #             color = clrs[j % n_lines]
    #             ax.plot(W_tril_norm, mean, label=r'W_' + str(j), c=color, alpha=0.7)
    #             ax.plot(W_tril_norm, mean, c=color, alpha=0.7, linewidth=1.5)
    #             ax.fill_between(W_tril_norm, mean - 2 * std, mean + 2 * std, alpha=0.2, facecolor=color)
    #             ax.plot(sklearn_norm, sklearn_sorted[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)
    #
    #         ax.set_xscale('log')
    #         plt.xlabel(r'$|W|/max(|W|)$', fontsize=18)
    #         plt.ylabel(r'$W$', fontsize=18)
    #         plt.xlim([W_tril_norm.min(), W_tril_norm.max()])
    #         plt.locator_params(axis='y', nbins=4)
    #         plt.xticks(fontsize=12)
    #         plt.yticks(fontsize=12)
    #         plt.savefig(f"{folder_name}W_norm_p{p.item():.2f}_T{T:.3f}_{i}.pdf", dpi=200, bbox_inches='tight')
    #         plt.close()

    return MSE_lambda

def plot_W_comparison(W_flow, W_sklearn, lambdas, p, T, conf=0.95, off_diagonal=True, folder_name=None, n_plots=3):

    if folder_name is None:
        folder_name = "./plots/"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    N = W_flow.shape[-1]
    if off_diagonal:
        indices = np.tril_indices(N, k=-1)
    else:
        indices = np.diag_indices(N)
    tril_indices = (..., indices[0], indices[1])
    W_tril = W_flow[tril_indices]
    W_tril_mean = W_tril.mean(1)
    W_tril_l = np.quantile(W_tril, 1-conf, axis=1)
    W_tril_r = np.quantile(W_tril, conf, axis=1)
    W_tril_sklearn = W_sklearn[tril_indices]

    MSE_lambda = np.sqrt(np.mean(np.square(W_tril_mean - W_tril_sklearn), 1))
    # print("MSE_lambda for each lambda", MSE_lambda)
    print("MSE_lambda mean", MSE_lambda.mean())

    # plot precision matrix entries as a function of lambda
    j = 0
    n_lines = W_tril_mean.shape[1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots()
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == W_tril_mean.shape[1]:
                    break
                color = clrs[j % n_lines]
                ax.plot(lambdas, W_tril_mean[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(lambdas, W_tril_l[:,j], W_tril_r[:,j], alpha=0.2, facecolor=color)
                ax.plot(lambdas, W_tril_sklearn[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            ax.set_xscale('log')
            plt.xlabel(r'$\lambda$', fontsize=18)
            plt.ylabel(r'$\Omega$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if off_diagonal:
                plt.savefig(f"{folder_name}W_lambda_p{p.item():.2f}_T{T:.3f}_off_{i}.pdf", bbox_inches='tight')
            else:
                plt.savefig(f"{folder_name}W_lambda_p{p.item():.2f}_T{T:.3f}_{i}.pdf", bbox_inches='tight')
            # plt.show()
            plt.close()

    # # compute norms
    # W_tril_norm = np.power(np.power(np.abs(W_tril_mean), p).sum(1), 1./p)
    # W_tril_norm /= W_tril_norm.max()
    # W_tril_norm_sorted_idx = W_tril_norm.argsort()
    # W_tril_norm = W_tril_norm[W_tril_norm_sorted_idx]
    # W_tril_norm_mean = W_tril_mean[W_tril_norm_sorted_idx]
    # W_tril_norm_std = W_tril_std[W_tril_norm_sorted_idx]
    #
    # sklearn_norm = np.power(np.power(np.abs(W_tril_sklearn), p).sum(1), 1./p)
    # sklearn_norm /= sklearn_norm.max()
    # sklearn_sorted_idx = sklearn_norm.argsort()
    # sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    # sklearn_sorted = W_tril_sklearn[sklearn_sorted_idx]
    #
    # for i in range(n_plots):
    #     fig, ax = plt.subplots()
    #     with sns.axes_style("darkgrid"):
    #         for j in range(i * n_lines, (i + 1) * n_lines):
    #             if j == W_tril_mean.shape[1]:
    #                 break
    #             std, mean = W_tril_norm_std[:, j], W_tril_norm_mean[:, j]
    #             color = clrs[j % n_lines]
    #             ax.plot(W_tril_norm, mean, label=r'W_' + str(j), c=color, alpha=0.7)
    #             ax.plot(W_tril_norm, mean, c=color, alpha=0.7, linewidth=1.5)
    #             ax.fill_between(W_tril_norm, mean - 2 * std, mean + 2 * std, alpha=0.2, facecolor=color)
    #             ax.plot(sklearn_norm, sklearn_sorted[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)
    #
    #         ax.set_xscale('log')
    #         plt.xlabel(r'$|W|/max(|W|)$', fontsize=18)
    #         plt.ylabel(r'$W$', fontsize=18)
    #         plt.xlim([W_tril_norm.min(), W_tril_norm.max()])
    #         plt.locator_params(axis='y', nbins=4)
    #         plt.xticks(fontsize=12)
    #         plt.yticks(fontsize=12)
    #         plt.savefig(f"{folder_name}W_norm_p{p.item():.2f}_T{T:.3f}_{i}.pdf", dpi=200, bbox_inches='tight')
    #         plt.close()

    return MSE_lambda

def plot_W_fixed_p (flow, S, p, T, lamb_min, lamb_max, X_train, n_plots=3, n_iter=50):
    n = X_train.shape[0]
    samples, kl_mean, kl_T_mean, kl_std, kl_T_std, W_mean, W_std, lambda_sorted = utils_mcf.sample_W_fixed_p(flow, S, p=p, n=n,
                                                                                                   T=T, context_size=2,
                                                                                                   sample_size=100,
                                                                                                   n_iter=n_iter,
                                                                                                   lambda_min_exp=lamb_min,
                                                                                                   lambda_max_exp=lamb_max)
    p_value = p.detach().cpu().numpy()

    alpha_sorted = lambda_sorted * 2 / n
    # W_sklearn = np.array([ADMM_SGL(S.detach().cpu().numpy(), lamb * 0.5, np.eye(S.shape[0]), max_iter=1000)[0]['Theta'] for lamb in tqdm.tqdm(alpha_sorted)])
    W_glasso = glasso_path(S.detach().cpu().numpy(), alpha_sorted * 0.5, diagonal=True)[1]  # (args.d, args.d, n_lambdas)
    W_glasso = np.moveaxis(W_glasso, 2, 0)  # (n_lambdas, args.d, args.d)
    # W_sklearn = np.array([graphical_lasso(emp_cov=S.detach().cpu().numpy(), alpha=lamb * 0.5)[1] for lamb in tqdm.tqdm(alpha_sorted)])

    # print('Diagonal elements')
    # MSE = plot_W_comparison(W_mean, W_std, W_sklearn, alpha_sorted, p=p_value, T=T, off_diagonal=False, n_plots=n_plots)
    print('Off-diagonal elements')
    MSE = plot_W_comparison(samples, W_glasso, alpha_sorted, p=p_value, T=T, off_diagonal=False, n_plots=3)
    MSE = plot_W_comparison(samples, W_glasso, alpha_sorted, p=p_value, T=T, off_diagonal=True, n_plots=n_plots)

    return MSE

def plot_W_bayes_lasso (S, lamb_min, lamb_max, X_train, n_lambdas=100, burnin = 1000, n_iter = 2000, n_plots=3):
    lambdas = np.logspace(lamb_min, lamb_max, n_lambdas)
    n = X_train.shape[0]
    d = X_train.shape[-1]
    samples = [bayes_lasso(S * n, n, lamb, n_iter, d, d)[0][burnin:].reshape(n_iter - burnin, d, d) for lamb in tqdm.tqdm(lambdas)]
    samples = np.array(samples)

    alpha_sorted = lambdas * 2 / n
    # W_sklearn = np.array([ADMM_SGL(S, lamb * 0.5, np.eye(S.shape[0]), max_iter=1000)[0]['Theta'] for lamb in tqdm.tqdm(alpha_sorted)])
    W_glasso = glasso_path(S, alpha_sorted * 0.5, diagonal=True)[1]  # (args.d, args.d, n_lambdas)
    W_glasso = np.moveaxis(W_glasso, 2, 0)  # (n_lambdas, args.d, args.d)
    print('Off-diagonal elements')
    MSE = plot_W_comparison(samples, W_glasso, alpha_sorted, p=np.array(1), T=1, off_diagonal=False, n_plots=n_plots, folder_name='./plots_bayes/')

    return MSE

def box_plot_comparison (S, lamb_min, lamb_max, X_train, p_min=0.25, p_max=1.25, epochs=2000, n_lambdas=10, burnin = 1000, n_iter = 2000, n_plots=3, seed=1234):
    lambdas = np.logspace(lamb_min, lamb_max, n_lambdas)
    n, d = X_train.shape
    # bayesian lasso
    samples_blasso = [bayes_lasso(S * n, n, lamb, n_iter, d, d)[0][burnin:].reshape(n_iter - burnin, d, d)[::10] for lamb in tqdm.tqdm(lambdas)]
    samples_blasso = np.array(samples_blasso)

    # lasso path
    # alpha_sorted = lambdas * 2 / n
    # samples_sklearn = np.array([ADMM_SGL(S, lamb * 0.5, np.eye(S.shape[0]), max_iter=1000)[0]['Theta'] for lamb in tqdm.tqdm(alpha_sorted)])
    W_glasso = glasso_path(S, lambdas * 2 / n * 0.5, diagonal=True)[1]  # (args.d, args.d, n_lambdas)
    W_glasso = np.moveaxis(W_glasso, 2, 0)  # (n_lambdas, args.d, args.d)

    # flow
    file_name = f'd{d}_n{n}_e{epochs}_pmin{p_min}_pmax{p_max}_lmin{lamb_min}_lmax{lamb_max}_seed{seed}_T1.000'
    flow = utils_mcf.build_positive_definite_vector(d, context_features=128, hidden_features=256, n_layers=10)
    flow.load_state_dict(torch.load(f"./models/cmf_{file_name}"))

    sample_size, context_size = 100, 100
    lambdas_exp = torch.from_numpy(lambdas).log10().view(-1, 1)
    lambdas_exp = lambdas_exp.cuda()
    samples_flow ={}
    for p_value in [1.0,0.75,0.5,0.25]:
        p = lambdas_exp.new_ones(1) * p_value
        uniform_p = lambdas_exp.new_ones(n_lambdas).view(-1, 1) * p
        context = torch.cat((lambdas_exp, uniform_p), 1).float()
        posterior_samples, _ = flow.sample_and_log_prob(sample_size, context=context)
        samples_flow[str(p_value)] = posterior_samples.detach().cpu().numpy()

    # create boxplot
    lambda_names = [f"{i:.1f}" for i in lambdas]
    for i in range(0,d,2):
        for j in range(i,d,2):
            plt.figure(figsize=(14,7))
            bayes_lasso_df = pd.DataFrame(samples_blasso[...,i,j].T, columns=lambda_names).assign(model="bayes lasso")
            flow_p100_df = pd.DataFrame(samples_flow['1.0'][...,i,j].T, columns=lambda_names).assign(model="flow (p=1.00)")
            flow_p075_df = pd.DataFrame(samples_flow['0.75'][...,i,j].T, columns=lambda_names).assign(model="flow (p=0.75)")
            flow_p050_df = pd.DataFrame(samples_flow['0.5'][...,i,j].T, columns=lambda_names).assign(model="flow (p=0.50)")
            flow_p025_df = pd.DataFrame(samples_flow['0.25'][...,i,j].T, columns=lambda_names).assign(model="flow (p=0.25)")
            cdf = pd.concat([bayes_lasso_df, flow_p100_df, flow_p075_df, flow_p050_df, flow_p025_df])
            mdf = pd.melt(cdf, id_vars=['model'], var_name=r'$\lambda$', value_name=r"$\beta$")
            sns.boxplot(x=r'$\lambda$', y=r"$\beta$", hue="model", data=mdf)
            plt.scatter(range(0,n_lambdas),W_glasso[...,i,j], marker='x', s=100, c='r', label='glasso')
            plt.legend()
            plt.savefig(f"./plots/box_plot_{i}_{j}.pdf", bbox_inches='tight')
            plt.close()
            plt.clf()

def plot_GLasso_solution(S, N, n, lamb_min_exp , lamb_max_exp, n_points = 100, solver='scikit'):
    lambdas = np.logspace(lamb_min_exp, lamb_max_exp, n_points)
    lambdas_idx = lambdas.argsort()
    lambdas = lambdas[lambdas_idx]
    alphas = lambdas * 2 / n
    if solver=='scikit':
        W_sklearn = np.array([graphical_lasso(emp_cov=S, alpha=lamb * 0.5, max_iter=1000)[1] for lamb in tqdm.tqdm(alphas)])
    elif solver== 'gglasso':
        W_sklearn = np.array([ADMM_SGL(S, lamb * 0.5, np.eye(N), max_iter=1000)[0]['Theta'] for lamb in tqdm.tqdm(alphas)])
    else:
        raise ValueError("solver must be one of either 'scikit' or 'gglasso'")
    indices = np.tril_indices(N, k=-1)
    tril_indices = (..., indices[0], indices[1])
    W_tril_sklearn = W_sklearn[tril_indices].reshape(n_points, -1)
    plt.plot(alphas, W_tril_sklearn)
    plt.xscale('log')
    plt.xlabel(r'$\lambda$', fontsize=18)
    plt.ylabel(r'$\Omega$', fontsize=18)
    plt.locator_params(axis='y', nbins=4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("glasso_solution.pdf", bbox_inches='tight')
    # plt.show()
    plt.clf()

    sklearn_norm = np.abs(W_tril_sklearn).sum(1)
    sklearn_norm /= sklearn_norm.max()
    sklearn_sorted_idx = sklearn_norm.argsort()
    sklearn_norm = sklearn_norm[sklearn_sorted_idx]
    sklearn_sorted = W_tril_sklearn[sklearn_sorted_idx]

    plt.plot(sklearn_norm, sklearn_sorted)
    plt.xscale('log')
    plt.xlabel(r'$|\Omega|/max(|\Omega|)$', fontsize=18)
    plt.ylabel(r'$\Omega$', fontsize=18)
    plt.locator_params(axis='y', nbins=4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("glasso_solution_norm.pdf", bbox_inches='tight')
    plt.clf()
    # plt.show()
