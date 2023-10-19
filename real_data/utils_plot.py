import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import os
import tqdm
import pyreadr
import utils_mcf
from gglasso.solver.single_admm_solver import ADMM_SGL
from sklearn.covariance import graphical_lasso, GraphicalLassoCV

# r = robjects.r
# r['source']('BayesLassoMCMC.R')
# glasso_path = robjects.globalenv['GLassoPath']
# bayes_lasso = robjects.globalenv['Bayes.glasso.MB']

# sns.set_style("whitegrid")
sns.set_theme(style="whitegrid")

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

def save_lambda(lambda_cmf, p, file_name=None, folder_name=None):
    if folder_name is None:
        folder_name = "./models/"

    if file_name is None:
        file_name = ''

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    f = open(f"{folder_name}optimal_lambda_cmf_{file_name}_p{p:.2f}.txt", "w")
    f.write(f"{lambda_cmf}")
    f.close()

def read_lambda(p, file_name=None, folder_name=None):
    f = open(f"{folder_name}optimal_lambda_cmf_{file_name}_p{p:.2f}.txt", "r")
    return f.read()

def plot_marginal_log_likelihood(lambda_sorted, kl, T, p, conf=0.95, file_name=None, folder_name=None):

    if folder_name is None:
        folder_name = "./plots/"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # compute credible intervals for marginal log likelihood
    # assuming convergence, the mll is equal to the negative kl divergence at the end of training
    mll_mean = -kl.mean(1)
    mll_l = np.quantile(-kl, 1 - conf, axis=1)
    mll_r = np.quantile(-kl, conf, axis=1)
    lambda_cmf = lambda_sorted[np.argmax(mll_mean)]  # lambda that maximises the marginal log likelihood

    fig, ax = plt.subplots()
    ax.plot(lambda_sorted, mll_mean, alpha=0.7)
    ax.fill_between(lambda_sorted, mll_l, mll_r, facecolor='b', alpha=0.3)
    plt.xscale('log')
    plt.xlabel(r'$\lambda$', fontsize=18)
    plt.ylabel(r'$\log p(X|\lambda)$', fontsize=18)
    plt.locator_params(axis='y', nbins=4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.vlines(lambda_cmf, mll_l.min(), mll_r.max(), label='CMF estimate', colors='b', linestyles='dotted')
    plt.legend()
    plt.savefig(f"{folder_name}W_mll_T{T}_{file_name}.pdf", bbox_inches='tight')
    plt.clf()
    # plt.show()
    print(f"optimal lambda CMF: {lambda_cmf}")

    save_lambda(lambda_cmf, p=p, file_name=file_name)

    return lambda_cmf


def plot_W_comparison(W_flow, W_glasso, lambda_sorted, lambda_glasso, T, p=1., conf=0.85, extract_triangular=True,
                      off_diagonal=True, folder_name=None, n_plots=3):

    if folder_name is None:
        folder_name = "./plots/"

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    N = W_flow.shape[-1]
    if extract_triangular:
        if off_diagonal:
            indices = np.tril_indices(N, k=-1)
        else:
            indices = np.diag_indices(N)
        tril_indices = (..., indices[0], indices[1])
        W_tril = W_flow[tril_indices].reshape(*W_flow.shape[:2], -1)
        W_tril_glasso = W_glasso[tril_indices].reshape(W_glasso.shape[0], -1)
    else:
        W_tril = W_flow.reshape(*W_flow.shape[:2], -1)
        W_tril_glasso = W_glasso.reshape(W_glasso.shape[0], -1)

    W_tril_mean = W_tril.mean(1)
    W_tril_l = np.quantile(W_tril, 1 - conf, axis=1)
    W_tril_r = np.quantile(W_tril, conf, axis=1)

    # plot precision matrix entries as a function of lambda
    n_lines = W_tril_mean.shape[1] // n_plots
    clrs = sns.color_palette("husl", n_lines)
    for i in range(n_plots):
        fig, ax = plt.subplots()
        with sns.axes_style("darkgrid"):
            for j in range(i * n_lines, (i + 1) * n_lines):
                if j == W_tril_mean.shape[1]:
                    break
                color = clrs[j % n_lines]
                ax.plot(lambda_sorted, W_tril_mean[:, j], c=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(lambda_sorted, W_tril_l[:,j], W_tril_r[:,j], alpha=0.2, facecolor=color)
                ax.plot(lambda_glasso, W_tril_glasso[:, j], linestyle='--', linewidth=1.5, c=color, alpha=0.7)

            #ax.set_xlim(min(alphas.min(), alphas_glasso.min()), max(alphas.max(), alphas_glasso.max()))
            ax.set_xscale('log')
            plt.xlabel(r'$\lambda$', fontsize=18)
            plt.ylabel(r'$\Omega$', fontsize=18)
            plt.locator_params(axis='y', nbins=4)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            if extract_triangular:
                plt.savefig(f"{folder_name}W_lambda_T{T:.3f}_p{p:.2f}_W11_{i}.png", dpi=200, bbox_inches='tight')
            else:
                plt.savefig(f"{folder_name}W_lambda_T{T:.3f}_p{p:.2f}_W12_{i}.png", dpi=200, bbox_inches='tight')
            plt.close()


def plot_full_comparison(model, S, P, Q, n, T, p, lambda_min, lambda_max, context_size=1, sample_size=500, n_iterations=100, file_name=None, plot_full_matrix=False, plot_mll=False):
    samples, kl, kl_T, W_mean, W_std, lambda_sorted = \
        utils_mcf.sample_W_fixed_p(model, S, P=P, Q=Q, n=n, T=T, p=p, context_size=context_size, sample_size=sample_size,
                         n_iterations=n_iterations, lambda_min_exp=lambda_min, lambda_max_exp=lambda_max)

    alpha_sorted = lambda_sorted * 2 / n
    if plot_mll:
        optimal_lambda = plot_marginal_log_likelihood(lambda_sorted, kl, T, p=p.item(), file_name=file_name)
    else:
        optimal_lambda = float(read_lambda(p.item(), file_name=file_name, folder_name="./models/"))
        # optimal_lambda = float(read_lambda(1.0, file_name=file_name, folder_name="./models/"))

    glasso_solution = utils_mcf.compute_glasso_solution(S, alpha_sorted)

    # plot W_11 block
    plot_W_comparison(samples[:, :, :P, :P], glasso_solution['W'][:, :P, :P], lambda_sorted=lambda_sorted,
                       lambda_glasso=glasso_solution['alphas'] * n / 2., T=T, p=p.item(), extract_triangular=True)

    # plot W_12 block
    plot_W_comparison(samples[:, :, :P, P:], glasso_solution['W'][:, :P, P:], lambda_sorted=lambda_sorted,
                       lambda_glasso=glasso_solution['alphas'] * n / 2., T=T, p=p.item(), extract_triangular=False, n_plots=20)

    samples = utils_mcf.sample_W_fixed_lamb_and_p(model, S, p=p.item(), exp_lamb=np.log10(optimal_lambda), n_iterations=50, sample_size=100)

    print("saving samples to RData...")
    # consider W_12 block only
    # breakpoint()
    if not plot_full_matrix:
        # N = samples.shape[-1]
        # indices = np.tril_indices(N, k=-1)
        # samples_W_11 = samples[:, indices[0], indices[1]]
        # samples_W_12 = samples[:, :, P:]
        # samples =
        samples = samples[:, :, P:]
    else:
        indices = np.diag_indices(P)
        samples[:,indices[0], indices[1]] = 0
    # ravel to vector column wise (R is column-major)
    samples = samples.reshape(samples.shape[0], -1, order='F')
    pyreadr.write_rdata(
        f"./cond_flow_data_optlamb_{optimal_lambda:.3f}_optalph_{optimal_lambda * 2 / n:.3f}_{T:.3f}_p{p.item():.2f}.RData",
        pd.DataFrame(samples), df_name="CMB.array", compress="gzip")


def plot_credibility_interval(p_values, file_name, n, n_values=5):

    P = 6
    Q = 312
    palette_tab10 = sns.color_palette("tab10", 10)
    palette_blue = list(sns.light_palette(palette_tab10[0], n_colors=6))[::-1][:5]
    palette_salmon = list(sns.light_palette(palette_tab10[1], n_colors=6))[::-1][:1]
    palette = palette_blue + palette_salmon

    assert p_values[0] == 1.0

    clin_vars = ['AGE', 'SEX', 'T', 'N', 'M', 'GS']
    gene_vars = list(np.load("column_names.npy", allow_pickle=True))
    clin_clin = np.array([f"({c},{g})" for i, c in enumerate(clin_vars) for g in clin_vars[i + 1:]])
    clin_gene = np.array([f"({c},{g})" for c in clin_vars for g in gene_vars])
    pair_vars = np.r_[clin_clin, clin_gene]

    samples_dict = {}
    for p_value in tqdm.tqdm(p_values):
        optimal_lambda = float(read_lambda(p=p_value, file_name=file_name, folder_name="./models/"))
        # file = f"./cond_flow_data_optlamb_28.537_optalph_0.293_1.000_p{p_value:.2f}.RData"
        file = f"./cond_flow_data_optlamb_{optimal_lambda:.3f}_optalph_{optimal_lambda * 2 / n:.3f}_1.000_p{p_value:.2f}.RData"
        samples = pyreadr.read_r(file)["CMB.array"].to_numpy(dtype=np.float32)
        samples = samples.reshape((-1, P, P+Q), order='F')
        indices = np.tril_indices(P, k=-1)
        samples_tril = samples[:,indices[0], indices[1]]
        samples_rect = samples[:,:,P:].reshape(-1, P * Q)
        samples_reshaped = np.c_[samples_tril, samples_rect]

        if p_value == 1.0:
            samples_median = np.median(samples_reshaped, axis=0)
            idx_median = np.argsort(np.abs(samples_median))[-n_values:][::-1]

        samples_dict[p_value] = samples_reshaped


    flow_samples = [pd.DataFrame(samples_dict[p_value][:,idx_median], columns=pair_vars[idx_median]).assign(model=f"CMF (q={p_value})")
                    for p_value in np.sort(p_values)]
    cdf = pd.concat(flow_samples)
    mdf = pd.melt(cdf, id_vars=['model'], var_name=r'(clin, gene)', value_name=r"$\Omega$")
    plt.figure(figsize=[2*6.4, 4.8])
    g1 = sns.boxplot(x=r'(clin, gene)', y=r"$\Omega$", hue="model", data=mdf, palette=palette, whis=[5, 95],
                showfliers=False)
    g1.set(xlabel=None)
    plt.locator_params(axis='y', nbins=4)
    # plt.xlabel(r'$\lambda$', fontsize=18)
    plt.ylabel(r'$\Omega$', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(f"./plots/box_plot.pdf", bbox_inches='tight')
    plt.close()
    plt.clf()