import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import BatchSGD, MomentumSGD, RMSPropMomentumSGD

# load data
data_path = os.path.join("data", "oil.csv")
data = pd.read_csv(data_path)
N = data['spills'].values
b1 = data['importexport'].values
b2 = data['domestic'].values
n_samples = len(N)


def log_likelihood(alpha: np.ndarray | list):
    """obejective function"""
    alpha = np.array(alpha)

    lam = alpha[0] * b1 + alpha[1] * b2
    return np.sum(N * np.log(lam) - lam)


def log_likelihood_grad(alpha: np.ndarray | list, batch_idx):
    """gradient function with batch index"""
    alpha = np.array(alpha)

    # batch sample
    lam = alpha[0] * b1[batch_idx] + alpha[1] * b2[batch_idx]
    g1 = np.sum(N[batch_idx] * b1[batch_idx] / lam - b1[batch_idx])
    g2 = np.sum(N[batch_idx] * b2[batch_idx] / lam - b2[batch_idx])

    # divide or not does not matter, => " lr v.s. lr / batch_size " whatever
    return np.array([g1, g2]) / len(batch_idx)
    # return np.array([g1, g2])  # lr -> lr / batch_size, similarly


def main():
    # random seed
    np.random.seed(42)

    # hype parameter
    lr = 0.011
    max_iter = 120
    batch_size = 1
    beta = 0.49
    rho = 0.85

    # start point
    start_point = [0.1, 0.1]

    # initial 3 optimizer
    batch_sgd = BatchSGD(
        f=log_likelihood,
        grad_f=log_likelihood_grad,
        lr=lr,
        batch_size=batch_size,
        epsilon=1e-6,
        max_iter=max_iter,
        repeat=False
    )

    momentum_sgd = MomentumSGD(
        beta=beta,
        f=log_likelihood,
        grad_f=log_likelihood_grad,
        lr=lr,
        batch_size=batch_size,
        epsilon=1e-6,
        max_iter=max_iter,
        repeat=False
    )

    rms_prop_momentum_sgd = RMSPropMomentumSGD(
        beta=beta,
        rho=rho,
        f=log_likelihood,
        grad_f=log_likelihood_grad,
        lr=lr,
        batch_size=batch_size,
        epsilon=1e-6,
        max_iter=max_iter,
        repeat=False
    )

    optimizers = [batch_sgd, momentum_sgd, rms_prop_momentum_sgd]
    for optimizer in optimizers:
        # 1. begin optimize
        optimizer.optimize(x0=start_point, n_samples=n_samples)

        # 2. summary to log
        optimizer.summary()

        # 3. save single figure
        figure_file_path = os.path.join(
            "figure", f"q4_{optimizer.__class__.__name__}.png"
        )
        optimizer.plot(
            figure_file_path=figure_file_path,
            title=optimizer.__class__.__name__,
            show=False
        )
        print("\n\n")

    # plot together
    plt.figure(figsize=(10, 6))  # compare figure
    for optimizer in optimizers:
        # get history, plot together
        history = optimizer.history()
        iters = [h["iteration"] for h in history]
        fs = [h["f_opt"] for h in history]

        plt.plot(iters, fs, label=optimizer.__class__.__name__)

    plt.xlabel("Iteration")
    plt.ylabel("f(x)")
    plt.title("Comparison of Optimization")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    compare_figure_path = os.path.join("figure", "q4_compare.png")
    plt.savefig(compare_figure_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {compare_figure_path}")


if __name__ == "__main__":
    # q4 (f) (g) (h)
    main()

    """
    python3 q4_fgh.py
    """
