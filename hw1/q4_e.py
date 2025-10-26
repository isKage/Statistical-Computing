import os
import math
import numpy as np
import pandas as pd

from utils import MultivariateQuasiNewtonOptimizer

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


def step_halving_or_not(lr: float = 1, step_halving: bool = True):
    """step_halving or not

    Args:
        step_halving (bool, optional): True -> applying step halving. Defaults to True.
    """
    # initial optimizer
    multivariate_quasi_newton_optimizer = MultivariateQuasiNewtonOptimizer(
        f=log_likelihood,
        grad_f=log_likelihood_grad,
        lr=lr,
        epsilon=1e-6,
        max_iter=300,
        step_halving=step_halving
    )

    # start point
    start_point = [
        [0.1, 0.1],
        [0.2, 0.2]
    ]
    for s in start_point:
        print(f"Start point: {s}")
        M0 = - information_matrix(s)

        # optimizer from M0
        result = multivariate_quasi_newton_optimizer.optimize(
            x0=s,
            M=M0
        )

        # summary
        multivariate_quasi_newton_optimizer.summary()

        # information matrix
        I = information_matrix(result["x_opt"])
        print(f"\nFisher information matrix at x_opt:\n{I}")

        # standard error
        se = np.sqrt(np.diag(np.linalg.inv(I)))
        print(f"\nStandard error:\n{se}\n\n")


def main():
    lr = 1
    step_halving_or_not(lr=lr, step_halving=True)
    step_halving_or_not(lr=lr, step_halving=False)


if __name__ == "__main__":
    # q4 (e)
    main()

    """
    python3 q4_e.py
    """
