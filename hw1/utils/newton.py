import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union


class NewtonOptimizer:
    """Newton Optimizer"""

    def __init__(self, f, f_prime=None, f_double_prime=None, epsilon: float = 1e-6, max_iter: int = 100, delta: float = 1e-6):
        """Initial Newton Optimizer class

        Args:
            f (function): objective function
            f_prime (function, optional): df/dx. if not known, defaults to None.
            f_double_prime (function, optional): d^2f/dx^2. if not known, defaults to None.
            epsilon (float, optional): precision. Defaults to 1e-6.
            max_iter (int, optional): max iteration rounds. Defaults to 100.
        """
        self.f = f
        self.f_prime = f_prime
        self.f_double_prime = f_double_prime
        self.epsilon = epsilon
        self.max_iter = max_iter

        self._start_point = None
        self._history = []  # store the history

        self.delta = delta  # if do not know derivative, applying numerical derivative

    def clear(self):
        """Clear history"""
        self._history = []

    def optimize(self, x0: float) -> dict:
        """Begin Optimize

        Args:
            x0 (float): initial x

        Returns:
            dict: result of the iteration
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

            # f_double_prime: f''
            if self.f_double_prime is not None:
                fpp = self.f_double_prime(x)
            else:
                fpp = self._numerical_derivative(x, 2)

            # f''(x) should large than 0, otherwise x - f'/f'' approx x
            if abs(fpp) < 1e-12:
                print(
                    f"Second derivative f''(x) is close to 0 at iteration {i + 1}. End!"
                )
                break

            # update
            x_new = x - fp / fpp

            # convergence or not
            if (abs(x_new - x) / (abs(x) + self.epsilon)) < self.epsilon:
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
            f"Newton's Method Optimization Process, starting from {round(self._start_point, 2)}"
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

        print("=" * 55)
        print(
            f"Coeffients:\nstart point = {self._start_point}\nepsilon = {self.epsilon}\nmax iteration = {self.max_iter}"
        )
        print("-" * 55)
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
        print("=" * 55)

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


class MultivariateNewtonOptimizer:
    """Multivariate Newton Optimizer"""

    def __init__(self, f, grad_f=None, hess_f=None, epsilon: float = 1e-6, max_iter: int = 100):
        """Initial Multivariate Newton Optimizer class

        Args:
            f (function): objective function
            grad_f (function): gradient function, returns np.array shape (p,)
            hess_f (function): Hessian function, returns np.array shape (p,p)
            epsilon (float, optional): precision. Defaults to 1e-6.
            max_iter (int, optional): max iteration rounds. Defaults to 100.
        """
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.epsilon = epsilon
        self.max_iter = max_iter

        self._start_point = None
        self._history = []

    def clear(self):
        """Clear history"""
        self._history = []

    def optimize(self, x0: Union[List[float], np.ndarray]) -> dict:
        """Begin Optimize

        Args:
            x0 (Union[List[float], np.ndarray]): start point

        Returns:
            dict: result
        """
        self._start_point = x0
        x = np.array(x0, dtype=float)

        # first point, note: x is array object, need to copy
        self._history = [
            {'iteration': 0, 'x_opt': x.copy(), 'f_opt': self.f(x)}]

        for i in range(self.max_iter):
            if self.grad_f is None or self.hess_f is None:
                grad = self._numerical_derivative(x, order=1)
                H = self._numerical_derivative(x, order=2)
            else:
                grad = self.grad_f(x)
                H = self.hess_f(x)

            # check H^{-1} exists or not
            if np.linalg.cond(H) > 1e12:
                # |H| |H^{-1}| > 1e12, ill-conditioned
                print(f"Hessian cannot be inversed at iter {i}. Early stop!")
                break

            # update: step = H^{-1} @ grad => H @ step = grad
            step = np.linalg.solve(H, grad)
            x_new = x - step

            # step = x - x_new, therefore: convergence check
            if np.linalg.norm(step) / (np.linalg.norm(x) + self.epsilon) < self.epsilon:
                x = x_new
                self._history.append({
                    'iteration': i+1,
                    'x_opt': x.copy(),
                    'f_opt': self.f(x)
                })
                break

            # go on
            x = x_new
            self._history.append(
                {'iteration': i+1, 'x_opt': x.copy(), 'f_opt': self.f(x)})

        return {
            'x_opt': x.copy(),
            'f_opt': self.f(x),
            'history': self._history.copy()
        }

    def summary(self):
        """Summary of this iteration"""
        if len(self._history) == 0:
            print("Must apply optimize() function first.")
            return None

        print("=" * 60)
        print(
            f"Coeffients:\nstart point = {self._start_point}\nepsilon = {self.epsilon}\nmax iteration = {self.max_iter}"
        )
        print("-" * 60)
        last_history = self._history[-1]
        print(
            f"x_opt = {last_history["x_opt"]}\nFuction(x_opt) = {last_history["f_opt"]}")

        # print iteration
        headers = ["Iter", "x", "", "f(x)", "|f(x*) - f(x)|"]
        print(
            f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12}"
        )
        print("-" * 60)
        for iter in self._history:
            print(f"{iter["iteration"]:<6} "
                  f"{iter["x_opt"][0]:<12.6f} "
                  f"{iter["x_opt"][1]:<12.6f} "
                  f"{iter["f_opt"]:<12.6f} "
                  f"{abs(last_history["f_opt"] - iter["f_opt"]):<12.6f}")
        print("=" * 60)

    def _numerical_derivative(self, x: Union[list, np.ndarray], order: int = 1) -> np.ndarray:
        pass
