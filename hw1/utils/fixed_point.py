import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union


class FixedPointOptimizer:
    """Fixed Point Optimizer"""

    def __init__(self, f, f_prime=None, alpha: float = 0.1, epsilon: float = 1e-6, max_iter: int = 100, delta: float = 1e-6):
        """Initial Fixed Point Optimizer

        Args:
            f (function): objective function
            f_prime (function, optional): df/dx. if not known, defaults to None.
            alpha (float, optional): learning rate. Defaults to 0.1.
            epsilon (float, optional): precision. Defaults to 1e-6.
            max_iter (int, optional): max number of iteration. Defaults to 100.
        """
        self.f = f
        self.f_prime = f_prime
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iter = max_iter

        self._history = []
        self._start_point = None

        self.delta = delta  # if do not know derivative, applying numerical derivative

    def clear(self):
        """clear the history"""
        self._history = []

    def optimize(self, x0: float) -> dict:
        """Begin optimize

        Args:
            x0 (float): start point

        Returns:
            dict: result dictionary
        """
        self.clear()

        self._start_point = x0
        x = x0
        self._history = [{
            'iteration': 0,
            'x_opt': x,
            'f_opt': self.f(x)
        }]

        for i in range(self.max_iter):
            # f_prime: f'
            if self.f_prime is not None:
                fp = self.f_prime(x)
            else:
                fp = self._numerical_derivative(x, 1)

            # update
            x_new = x + self.alpha * fp

            # convergence
            if abs(x_new - x) / (abs(x) + self.epsilon) < self.epsilon:
                x = x_new
                self._history.append({
                    'iteration': i + 1,
                    'x_opt': x,
                    'f_opt': self.f(x)
                })
                break
            # go on
            x = x_new
            self._history.append({
                'iteration': i + 1,
                'x_opt': x,
                'f_opt': self.f(x)
            })

        # result
        result = {
            'x_opt': x,
            'f_opt': self.f(x),
            'history': self._history.copy()
        }
        return result

    def plot(self, figure_file_path: str | None = None, x_range: Tuple[float, float] = None, show: bool = False, num_points: int = 1000):
        """plot function curve

        Args:
            figure_file_path (str | None, optional): saved path
            x_range (Tuple[float, float], optional): x's interval. Defaults to None.
            num_points (int, optional): number of points to plot. Defaults to 1000.

        Returns:
            None: None
        """
        if len(self._history) == 0:
            print("Must apply optimize() function first.")
            return None

        # iteration values
        x_opt = [point["x_opt"] for point in self._history]
        f_opt = [point["f_opt"] for point in self._history]

        # create figure
        plt.figure(figsize=(10, 6))

        if x_range is None:
            # new x range
            x_min, x_max = min(x_opt), max(x_opt)
            x_padding = 0.1 * (x_max - x_min)  # expand the range
            x_plot = np.linspace(
                x_min - x_padding, x_max + x_padding, num_points)
        else:
            # we set the x range
            x_plot = np.linspace(x_range[0], x_range[1], num_points)
        # f(x) value
        f_plot = [self.f(x) for x in x_plot]

        # plot the function curve
        plt.plot(x_plot, f_plot, 'b-', linewidth=1, label='Objective function')

        # plot iteration opt value
        plt.plot(x_opt, f_opt, 'ro-', linewidth=0.75,
                 markersize=2, label='Optimization path')

        # mark starting point
        plt.plot(x_opt[0], f_opt[0], 'go', markersize=4, label='Start point')

        # mark final optimum point
        plt.plot(x_opt[-1], f_opt[-1], 'rs',
                 markersize=4, label='Opt point')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(
            f"Fixed Point Method Optimization Process, starting from {round(self._start_point, 2)}, with alpha = {round(self.alpha, 2)}"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save figure
        if figure_file_path:
            plt.savefig(figure_file_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {figure_file_path}")

        if show:
            plt.show()

    def summary(self):
        """Summary of this iteration"""
        if len(self._history) == 0:
            print("Must apply optimize() function first.")
            return None

        print("="*55)
        print(
            f"Coeffients:\nstart point = {self._start_point}\nalpha = {self.alpha}\nepsilon = {self.epsilon}\nmax iteration = {self.max_iter}"
        )
        print("-"*55)
        last_history = self._history[-1]
        print(
            f"x_opt = {last_history["x_opt"]}\nFuction(x_opt) = {last_history["f_opt"]}")

        # print iteration
        headers = ["Iter", "x", "f(x)", "|f(x*) - f(x)|"]
        print(
            f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12}"
        )
        print("-" * 55)
        for iter in self._history:
            print(f"{iter["iteration"]:<6} "
                  f"{iter["x_opt"]:<12.6f} "
                  f"{iter["f_opt"]:<12.6f} "
                  f"{abs(last_history["f_opt"] - iter["f_opt"]):<12.6f}")
        print("="*55)

    def _numerical_derivative(self, x: float, order: int = 1) -> float:
        """numerical derivative

        Args:
            x (float): independent variables (uni-variable)
            order (int, optional): the number of prime. Defaults to 1.

        Raises:
            ValueError: only consider 1 and 2 prime

        Returns:
            float: derivative value
        """
        if order == 1:
            return (self.f(x + self.delta) - self.f(x - self.delta)) / (2 * self.delta)
        elif order == 2:
            return (self.f(x + self.delta) - 2 * self.f(x) + self.f(x - self.delta)) / (self.delta ** 2)
        else:
            raise ValueError("Order must be 1 or 2")
