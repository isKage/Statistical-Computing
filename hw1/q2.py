import os
import math
import numpy as np

from utils import NewtonOptimizer

x_sample = np.array([
    3.91, 4.85, 2.28, 4.06, 3.70, 4.04, 5.46, 3.53, 2.28, 1.96,
    2.53, 3.88, 2.22, 3.47, 4.82, 2.46, 2.99, 2.54, 0.52, 2.50
])
n = len(x_sample)


def log_likelihood_function(theta):
    """objective function"""
    if theta is None:
        return None  # in case of theta = None
    total = 0
    for x in x_sample:
        total += np.log(1 - np.cos(x - theta))
    return total - n * np.log(2 * np.pi)


def log_likelihood_function_prime(theta):
    """f prime"""
    if theta is None:
        return None  # in case of theta = None
    total = 0
    for x in x_sample:
        total += np.sin(x - theta) / (1 - np.cos(x - theta))
    return -total


def log_likelihood_function_double_prime(theta):
    """f double prime"""
    if theta is None:
        return None  # in case of theta = None
    total = 0
    for x in x_sample:
        total += 1 / (1 - np.cos(x - theta))
    return -total


def main():
    # Moment estimation
    theta_moment_estimate = np.arcsin(x_sample.mean() - np.pi)

    # Newton method
    newton_optimizer = NewtonOptimizer(
        f=log_likelihood_function,
        f_prime=log_likelihood_function_prime,
        f_double_prime=log_likelihood_function_double_prime,
        epsilon=1e-6,
        max_iter=50
    )

    # different start points
    start_points = [theta_moment_estimate, -2.7, 2.7]
    for i, start in enumerate(start_points):
        newton_optimizer.optimize(x0=start)

        figure_file_path = os.path.join("figure", f"q2_{i+1}.png")
        newton_optimizer.plot(
            figure_file_path=figure_file_path, x_range=[-np.pi, np.pi]
        )
        newton_optimizer.summary()
        print(
            f"Moment Estimation = {theta_moment_estimate}\nL(Moment Estimation) = {log_likelihood_function(theta_moment_estimate)}"
        )
        print("\n"*2)
        newton_optimizer.clear()


if __name__ == "__main__":
    main()

    """
    python3 q2.py
    """
