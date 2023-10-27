from scipy import stats
import numpy as np
import scipy as sp
import torch
from sklearn.datasets import make_sparse_spd_matrix
import utils_mcf, utils_plot
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--d', metavar='d', type=int, default=30,
                    help='number of features/nodes')
parser.add_argument('--n', metavar='n', type=int, default=100,
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

    # generate precision matrix:
    W = make_sparse_spd_matrix(args.d, alpha=0.9, random_state=2)

    # generate observations from gaussian distribution
    cov = sp.linalg.inv(W)
    distr = sp.stats.multivariate_normal(mean=np.zeros(args.d), cov=cov)
    X = distr.rvs(100_000)[:args.n]
    # X -= X.mean(0)
    X /= X.std(0)
    S = np.cov(X, rowvar=False)


    # Build Conditional Matrix Flow
    S_torch = torch.from_numpy(S).float().cuda()
    matrix_dim = S_torch.shape[0]
    flow = utils_mcf.build_positive_definite_vector(matrix_dim, context_features=32, n_layers=3)

    # train model
    p_min, p_max = .1, 1.5
    T0, Tn = 5, 1
    flow, loss, loss_T = utils_mcf.train_model(flow, S_torch, X, d=args.d, n=args.n, lr=1e-3, epochs=args.epochs, context_size=1_000, p_min=p_min, p_max=p_max,
                                     lambda_min_exp=lamb_min_exp, lambda_max_exp=lamb_max_exp, T0=T0, Tn=Tn, iter_per_cool_step=50, seed=args.seed)


if __name__ == "__main__":
    main()
