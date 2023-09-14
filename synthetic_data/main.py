from scipy import stats
import numpy as np
import scipy as sp
import torch
from sklearn.datasets import make_sparse_spd_matrix
import utils_mcf, utils_plot
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--d', metavar='d', type=int, default=15,
                    help='number of features/nodes')
parser.add_argument('--n', metavar='n', type=int, default=15,
                    help='number of samples')
parser.add_argument('--epochs', metavar='e', type=int, default=10000,
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

    # generate precision matrix:
    W = make_sparse_spd_matrix(args.d, alpha=0.9, random_state=2)

    # generate observations from gaussian distribution
    cov = sp.linalg.inv(W)
    distr = sp.stats.multivariate_normal(mean=np.zeros(args.d), cov=cov)
    X = distr.rvs(100_000)[:args.n]
    # X -= X.mean(0)
    X /= X.std(0)
    S = np.cov(X, rowvar=False)

    # plot ground truth precision matrix W
    # plot_W_gt(W)

    # compute and plot glasso estimate of precision matrix W
    # prec_gl = utils_plot.glasso_solution(S, W, alpha=1e1 * 2 / args.n)

    # check lambda range by plotting GLasso solution
    lamb_min_exp, lamb_max_exp = 0, 2
    utils_plot.plot_GLasso_solution(S, args.d, args.n, lamb_min_exp, lamb_max_exp, n_points=100, solver='gglasso')

    # n_plots = 20
    # MSE_bayes_lasso = utils_plot.plot_W_bayes_lasso(S, lamb_min=lamb_min_exp, lamb_max=lamb_max_exp, X_train=X, n_lambdas=100, n_plots=n_plots)
    # breakpoint()

    # Build Conditional Matrix Flow
    S_torch = torch.from_numpy(S).float().cuda()
    matrix_dim = S_torch.shape[0]
    flow = utils_mcf.build_positive_definite_vector(matrix_dim, context_features=128, hidden_features=256, n_layers=10)

    # train model
    p_min, p_max = .25, 1.25
    T0, Tn = 10, 1e-3
    flow, loss, loss_T = utils_mcf.train_model(flow, S_torch, X, d=args.d, n=args.n, lr=1e-3, epochs=args.epochs, context_size=1_000, p_min=p_min, p_max=p_max,
                                               lambda_min_exp=lamb_min_exp, lambda_max_exp=lamb_max_exp, T0=T0, Tn=Tn, iter_per_cool_step=100, seed=args.seed)

    # plot loss function
    utils_plot.plot_loss(loss, loss_T)

    # plot MAP for different l_p norms
    n_plots = 10
    p = torch.tensor(1.).cuda() # Lasso solution
    MSE_flow = utils_plot.plot_W_fixed_p(flow, S_torch, p=p, T=Tn, lamb_min=lamb_min_exp, lamb_max=lamb_max_exp, X_train=X, n_plots=n_plots)


    sub_l1 = [0.75, 0.5, 0.25] # sub-l1 pseudo-norms
    for i in sub_l1:
        print(f"p = {i}")
        utils_plot.plot_W_fixed_p(flow, S_torch, p=p * i, T=Tn, lamb_min=lamb_min_exp, lamb_max=lamb_max_exp, X_train=X, n_plots=n_plots)

    utils_plot.box_plot_comparison(S, lamb_min_exp, lamb_max_exp, X, p_min=p_min, p_max=p_max, epochs=5000, n_lambdas=10, burnin=1000,
                        n_iter=2000, n_plots=3, seed=args.seed)


if __name__ == "__main__":
    main()
