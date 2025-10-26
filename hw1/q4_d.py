import os
import math
import numpy as np
import pandas as pd

from utils import GradientAscentOptimizer

# load data
data_path = os.path.join("data", "oil.csv")
data = pd.read_csv(data_path)
N = data['spills'].values
b1 = data['importexport'].values
b2 = data['domestic'].values


def log_likelihood(alpha: np.ndarray | list):
    """obejective function"""
    alpha = np.array(alpha)

    lam = alpha[0] * b1 + alpha[1] * b2
    if np.any(lam <= 0):
        return -np.inf
    return np.sum(N * np.log(lam) - lam)


def log_likelihood_grad(alpha: np.ndarray | list):
    """gradient of objective function"""
    alpha = np.array(alpha)

    lam = alpha[0] * b1 + alpha[1] * b2
    g1 = np.sum(N * b1 / lam - b1)
    g2 = np.sum(N * b2 / lam - b2)
    return np.array([g1, g2])


def log_likelihood_hess(alpha: np.ndarray | list):
    """Hessian of objective function"""
    alpha = np.array(alpha)

    lam = alpha[0]*b1 + alpha[1]*b2
    h11 = - np.sum(N * b1**2 / lam**2)
    h22 = - np.sum(N * b2**2 / lam**2)
    h12 = - np.sum(N * b1*b2 / lam**2)
    return np.array([[h11, h12], [h12, h22]])


def information_matrix(alpha: np.ndarray | list):
    """Fisher information matrix"""
    alpha = np.array(alpha)

    lam = alpha[0] * b1 + alpha[1] * b2
    i11 = np.sum(b1**2 / lam)
    i22 = np.sum(b2**2 / lam)
    i12 = np.sum(b1 * b2 / lam)
    return np.array([[i11, i12], [i12, i22]])


def main():
    # initial optimizer
    gradient_ascent_optimizer = GradientAscentOptimizer(
        f=log_likelihood,
        grad_f=log_likelihood_grad,
        lr=1,
        epsilon=1e-6,
        max_iter=300
    )

    # optimizer
    result = gradient_ascent_optimizer.optimize(x0=[0.1, 0.1])

    # summary
    gradient_ascent_optimizer.summary()

    # information matrix
    I = information_matrix(result["x_opt"])
    print(f"\nFisher information matrix at x_opt:\n{I}")

    # standard error
    se = np.sqrt(np.diag(np.linalg.inv(I)))
    print(f"\nStandard error:\n{se}")


if __name__ == "__main__":
    # q4 (d)
    main()

    """
    python3 q4_d.py
    """
