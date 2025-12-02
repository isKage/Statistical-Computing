import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats

LOG_NUM = 50
N = 100
np.random.seed(42)


def q(x):
    return np.exp(-np.abs(x) ** 3 / 3.0)


def h(x):
    return x**2


def SIR():
    """SIR: Sampling Importance Resampling, estimation do not need `Resampling`"""
    # hypeparameters
    b = 1.0
    print("=" * LOG_NUM + "\nSIR: Sampling Importance Resampling")
    print(f"Sampling Num: {N}")

    # 1. sampling
    x = np.random.normal(loc=0.0, scale=b, size=N)

    # 2. importance rate
    qx = q(x)
    gx = stats.norm.pdf(x, loc=0.0, scale=b)

    # weight and normalized the weight
    w_star = qx / gx
    w = w_star / np.sum(w_star)

    # 3. estimate
    sigma_hat = np.sum(w * h(x))

    # result
    print(f"\n# Sigma Prediction: {sigma_hat:.6f}\n" + "=" * LOG_NUM)


def RS():
    """RS: Rejection Sampling"""
    # hypeparameters
    x0 = 1.5
    alpha = 3 * x0**2 / (4 * x0**3 + 6)
    print("=" * LOG_NUM + "\nRS: Rejection Sampling")
    print(f"Sampling Num: {N}")
    print(f"Envelope use x0 = {x0:.2f}, alpha = {alpha:.2f}")

    # 1. sampling u
    u = np.random.uniform(0, 1, N)

    # 2. transfer to x
    def G_inverse(u):
        """vector function"""
        cond1 = u >= 1 - alpha / x0**2
        cond2 = u <= alpha / x0**2
        cond3 = (~cond1) & (~cond2)

        x = np.zeros_like(u)
        x[cond1] = 2 * x0 / 3 - np.log((1 - u[cond1]) * x0**2 / alpha) / x0**2
        x[cond2] = np.log(u[cond2] * x0**2 / alpha) / x0**2 - 2 * x0 / 3
        x[cond3] = (u[cond3] - 0.5) / alpha
        return x

    x = G_inverse(u)  # X sim g(X)

    # 3. calculate the threshold
    qx = q(x)

    def e(x):
        """vector function"""
        cond1 = x > 2 * x0**2 / 3
        cond2 = x < -2 * x0**2 / 3
        cond3 = (~cond1) & (~cond2)

        ex = np.zeros_like(x)
        ex[cond1] = np.exp(- x0**2 * x[cond1] + 2 * x0**2 / 3)
        ex[cond2] = np.exp(x0**2 * x[cond2] + 2 * x0**2 / 3)
        ex[cond3] = 1
        return ex

    ex = e(x)
    thresholds = qx / ex
    print(f"{np.sum(thresholds > 1)} samples: q(X)/e(X) > 1")

    # 4. rejection
    z = np.random.uniform(0, 1, N)
    accept = z < thresholds
    x_star = x[accept]
    print(f"Accept rate = {x_star.size / N:.2%}")

    # 5. estimation with x*
    sigma_hat = np.mean(h(x_star))

    # result
    print(f"\n# Sigma Prediction: {sigma_hat:.6f}\n" + "=" * LOG_NUM)


if __name__ == "__main__":
    SIR()
    print()
    RS()
