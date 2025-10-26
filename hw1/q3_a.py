import os
import math
import numpy as np

from utils import FixedPointOptimizer

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


def establish_optimize(alpha: float) -> FixedPointOptimizer:
    """Establish Fixed Point Optimizer with different learning rate

    Args:
        alpha (float): learning rate

    Returns:
        FixedPointOptimizer: FixedPointOptimizer class
    """
    fixed_point_optimizer = FixedPointOptimizer(
        f=log_likelihood_function,
        f_prime=log_likelihood_function_prime,
        alpha=alpha,
        epsilon=1e-6,
        max_iter=50
    )
    return fixed_point_optimizer


def main():
    alpha_list = [1, 0.64, 0.25]
    for i, alpha in enumerate(alpha_list):
        # Fixed Point method
        fixed_point_optimizer = establish_optimize(alpha=alpha)

        # different start points
        start_points = [-1, 4]
        for j, start in enumerate(start_points):
            fixed_point_optimizer.optimize(x0=start)

            figure_file_path = os.path.join(
                "figure", f"q3_fixed_point_{i+1}_{j+1}.png"
            )
            fixed_point_optimizer.plot(
                figure_file_path=figure_file_path, x_range=[-5, 5]
            )
            fixed_point_optimizer.summary()
            print("\n"*2)
            fixed_point_optimizer.clear()


if __name__ == "__main__":
    main()

    """
    python3 q3_a.py
    """
