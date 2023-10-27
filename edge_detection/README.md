# Conditional Matrix Flows

This repository is built on the [nflows package](https://github.com/bayesiains/nflows).
`nflows` is a comprehensive collection of [normalizing flows](https://arxiv.org/abs/1912.02762)
using [PyTorch](https://pytorch.org).

### Layers

Main addition compared to the base repository include:
1. The proposed Sum-of-Sigmoids layers:
   - basic implementation in `nflows.transforms.adaptive_sigmoids` (SumofSigmoids)
   - autoregressive version in `nflows.transforms.autoregressive` (MaskedSumofSigmoids)
2. The proposed transformation to symmetric positive-definite matrices
   - FillTriangular, TransformDiagonalSoftplus, CholeskyOuterProduct in `nflows.transforms.cholesky` and 
     `nflows.transforms.matrix_transform`

### Experiments
We provide the code to reproduce the results illustrated in the rebuttal
- 'main.py' performs training for the proposed Conditional Matrix Flow
- 'main_scirpt.py' prints the commands for running 'main.py' with the hyper-parameters used in the experiments
- 'results.ipynb' contains the code to generate the plots of the rebuttal (it requires CMFs trained from 'main.py')

### Development
To install all the dependencies, take a look at the conda environment provided in `env`.
