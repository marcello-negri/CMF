from scipy import stats
import numpy as np
import scipy as sp
import torch
import os
from sklearn.datasets import make_sparse_spd_matrix
import utils_mcf, utils_plot
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--d', metavar='d', type=int, default=15,
                    help='number of features/nodes')
parser.add_argument('--n', metavar='n', type=int, default=15,
                    help='number of samples')
parser.add_argument('--epochs', metavar='e', type=int, default=10_000,
                    help='number of epochs')
parser.add_argument('--seed', metavar='s', type=int, default=1,
                    help='random seed')
parser.add_argument('--p_min', metavar='p_min', type=float, default=0.25,
                    help='p min')
parser.add_argument('--p_max', metavar='p_max', type=float, default=1.25,
                    help='p max')
parser.add_argument('--lambda_min', metavar='l_min', type=float, default=-0.5,
                    help='lambda min')
parser.add_argument('--lambda_max', metavar='l_max', type=float, default=1.,
                    help='lambda max')
parser.add_argument('--T0', metavar='T0', type=float, default=5.,
                    help='initial temperature')
parser.add_argument('--Tn', metavar='Tn', type=float, default=1e-2,
                    help='final temperature')

args = parser.parse_args()

def set_random_seeds (seed=1234):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    # set random seed for reproducibility
    set_random_seeds(args.seed)

    # generate precision matrix:
    W = make_sparse_spd_matrix(args.d, alpha=0.9, random_state=2, smallest_coef=0.1, largest_coef=0.9)

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
    utils_plot.plot_GLasso_solution(S, args.d, args.n, args.lambda_min, args.lambda_max, n_points=100, solver='gglasso')

    # Build Conditional Matrix Flow
    S_torch = torch.from_numpy(S).float().cuda()
    matrix_dim = S_torch.shape[0]
    flow = utils_mcf.build_positive_definite_vector(matrix_dim, context_features=64, hidden_features=256, n_layers=10)

    # train model
    file_name = f'd{args.d}_n{args.n}_e{args.epochs}_pmin{args.p_min}_pmax{args.p_max}_lmin{args.lambda_min}_lmax{args.lambda_max}_seed{args.seed}_T{args.Tn:.3f}'
    if os.path.isfile(f"./models/cmf_{file_name}"):
        flow.load_state_dict(torch.load(f"./models/cmf_{file_name}"))
    else:
        flow, loss, loss_T = utils_mcf.train_model(flow, S_torch, X, d=args.d, n=args.n, lr=1e-3, epochs=args.epochs, context_size=1_000, p_min=args.p_min, p_max=args.p_max,
                                                   lambda_min_exp=args.lambda_min, lambda_max_exp=args.lambda_max, T0=args.T0, Tn=args.Tn, iter_per_cool_step=100, seed=args.seed)
        # plot loss function
        utils_plot.plot_loss(loss, loss_T)

    if args.Tn == 1e-2: # plot MAP for different l_p norms
        n_plots = 10
        p = torch.tensor(1.).cuda() # Lasso solution
        sub_l1 = [1.0, 0.75, 0.5, 0.25] # sub-l1 pseudo-norms
        for i in sub_l1:
            print(f"p = {i}")
            utils_plot.plot_W_fixed_p(flow, S_torch, p=p * i, T=args.Tn, lamb_min=args.lambda_min, lamb_max=args.lambda_max, X_train=X, n_plots=n_plots, n_iter=200, sample_size=500)
    elif args.Tn == 1.:
        utils_plot.box_plot_comparison(S, args.lambda_min, args.lambda_max, X, p_min=args.p_min, p_max=args.p_max, epochs=args.epochs, n_lambdas=4, burnin=1000,
                                       n_iter=2000, n_plots=3, seed=args.seed)


if __name__ == "__main__":
    main()
