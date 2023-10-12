from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.datasets import make_sparse_spd_matrix
import utils_mcf, utils_plot
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--d', metavar='d', type=int, default=15,
                    help='number of features/nodes')
parser.add_argument('--n', metavar='n', type=int, default=15,
                    help='number of samples')
parser.add_argument('--epochs', metavar='e', type=int, default=5_000,
                    help='number of epochs')
parser.add_argument('--seed', metavar='s', type=int, default=1234,
                    help='random seed')

args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)

    # load dataset
    df_X = pd.read_csv('./data/preprocessed_data.csv')
    X = df_X.to_numpy()

    # standardize dataset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n = X.shape[0]  # number of patients
    N = X.shape[1]  # total number of features
    P = 6  # clinical variables
    Q = N - P  # gene expression measurements
    print(f"The data consists of {P} clinical variables and {Q} gene expression measurements for {n} patients")
    S = np.cov(X, rowvar=False)

    # Flow Variational Inference for Bayesian GLASSO
    S_torch = torch.from_numpy(S).float().cuda()
    flow = utils_mcf.build_cond_psd_uncontrained_vector(P, Q, context_features=64, n_layers=4)

    # train model
    lamb_min, lamb_max = 1, 3
    p_min, p_max = 1,1
    T0, Tn = 5, 1
    flow, loss, loss_T = utils_mcf.train_model(flow, S_torch, P, Q, n, p_min=p_min, p_max=p_max, lr=1e-3, epochs=15_001, context_size=300,
                                     lambda_min_exp=lamb_min, lambda_max_exp=lamb_max, T0=T0, Tn=Tn, iter_per_cool_step=100)

    # sub_l1 = [0.75, 0.5, 0.25]  # sub-l1 pseudo-norms
    # for i in sub_l1:
    #     print(f"p = {i}")
    #     p = S_torch.new_ones(1)
    #     samples, kl, kl_T, W_mean, W_std, lambda_sorted = \
    #        utils_mcf.sample_W_fixed_p(flow, S, P=P, Q=Q, n=n, T=Tn, p=p, context_size=2, sample_size=50,
    #                          n_iterations=50, lambda_min_exp=lamb_min, lambda_max_exp=lamb_max)
    #     glasso_solution = utils_mcf.compute_glasso_solution(S, lambda_sorted * 2 / n)
    #
    #     # plot W_11 block
    #     MSE_W11 = utils_plot.plot_W_comparison(samples[:, :, :P, :P], glasso_solution['W'][:, :P, :P],
    #                                            lambda_sorted=lambda_sorted,
    #                                            lambda_glasso=glasso_solution['alphas'] * n / 2., T=Tn,
    #                                            extract_triangular=True)
    #
    #     # plot W_12 block
    #     MSE_W12 = utils_plot.plot_W_comparison(samples[:, :, :P, P:], glasso_solution['W'][:, :P, P:],
    #                                            lambda_sorted=lambda_sorted,
    #                                            lambda_glasso=glasso_solution['alphas'] * n / 2.,
    #                                            T=Tn, extract_triangular=False, n_plots=20)
    breakpoint()


if __name__ == "__main__":
    main()
