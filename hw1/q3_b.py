import os
import math
import numpy as np

from utils import SecantOptimizer

x_sample = np.array([
    1.77, -0.23, 2.76, 3.80, 3.47, 56.75, -1.34, 4.24, -2.44, 3.29,
    3.71, -2.40, 4.53, -0.07, -1.05, -13.87, -2.53, -1.75, 0.27, 43.21
])
n = len(x_sample)


def log_likelihood_function(theta):
    """objective function"""
    if theta is None:
        return None  # in case of theta = None
    total = 0
    for x in x_sample:
        total += np.log(1 + (x - theta)**2)
    return -total - n * np.log(np.pi)


def log_likelihood_function_prime(theta):
    """f prime"""
    if theta is None:
        return None  # in case of theta = None
    total = 0
    for x in x_sample:
        total += (x - theta) / (1 + (x - theta)**2)
    return 2 * total


def main():
    # Secant Optimizer
    secant_optimizer = SecantOptimizer(
        f=log_likelihood_function,
        f_prime=log_likelihood_function_prime,
        epsilon=1e-6,
        max_iter=50
    )

    # different start points
    start_points = [(-2, -1), (2, 1)]
    for i, start_point in enumerate(start_points):
        secant_optimizer.optimize(x0=start_point[0], x1=start_point[1])

        figure_file_path = os.path.join(
            "figure", f"q3_secant_{i+1}.png"
        )
        secant_optimizer.plot(
            figure_file_path=figure_file_path, x_range=[-5, 5]
        )
        secant_optimizer.summary()
        print("\n"*2)
        secant_optimizer.clear()


if __name__ == "__main__":
    main()

    """
    python3 q3_b.py
    """
