import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union


class GradientAscentOptimizer:
    """Gradient Ascent Optimizer with step-halving"""

    def __init__(self, f, grad_f=None, lr: float = 1, epsilon: float = 1e-6, max_iter: int = 100):
        """Initialize Gradient Ascent Optimizer

        Args:
            f (function): objective function
            grad_f (function): gradient function, returns np.array shape (p,)
            lr (float, optional): initial learning rate. Defaults to 1.
            epsilon (float, optional): precision. Defaults to 1e-6.
            max_iter (int, optional): max iteration count. Defaults to 100.
        """
        self.f = f
        self.grad_f = grad_f
        self.lr = lr
        self.epsilon = epsilon
        self.max_iter = max_iter

        self._start_point = None
        self._history = []

    def clear(self):
        """Clear iteration history"""
        self._history = []

    def optimize(self, x0: Union[List[float], np.ndarray]) -> dict:
        """Begin gradient ascent optimization with step-halving

        Args:
            x0 (Union[List[float], np.ndarray]): start point

        Returns:
            dict: result
        """
        x = np.array(x0, dtype=float)

        self._start_point = x.copy()
        self._history = [
            {'iteration': 0, 'x_opt': x.copy(), 'f_opt': self.f(x), 'lr': self.lr}
        ]

        # previous f value
        prev_f = self.f(x)

        for i in range(self.max_iter):
            lr = self.lr  # replace learning rate

            # gradient
            if self.grad_f is not None:
                grad = self.grad_f(x)
            else:
                grad = self._numerical_derivative(x)

            # if np.linalg.norm(grad) < self.epsilon:
            #     break

            # update (up)
            x_new = x + lr * grad
            f_new = self.f(x_new)

            # update learning rate
            while f_new <= prev_f:  # prev better => step-halving
                lr = lr / 2
                x_new = x + lr * grad
                f_new = self.f(x_new)

            # convergence check
            if (np.linalg.norm(x_new - x) / (np.linalg.norm(x) + self.epsilon)) < self.epsilon:
                x = x_new
                prev_f = f_new
                self._history.append(
                    {'iteration': i + 1, 'x_opt': x.copy(), 'f_opt': f_new, 'lr': lr}
                )
                break

            # go on
            x, prev_f = x_new, f_new
            self._history.append(
                {'iteration': i + 1, 'x_opt': x.copy(), 'f_opt': f_new, 'lr': lr}
            )

        return {
            'x_opt': x.copy(),
            'f_opt': self.f(x),
            'lr': lr,
            'history': self._history.copy()
        }

    def summary(self):
        """Summary of this iteration"""
        if len(self._history) == 0:
            print("Must apply optimize() function first.")
            return None

        print("=" * 75)
        print(
            f"Coeffients:\nstart point = {self._start_point}\ninitial learning rate = {self.lr}\nepsilon = {self.epsilon}\nmax iteration = {self.max_iter}"
        )
        print("-" * 75)
        last_history = self._history[-1]
        print(
            f"x_opt = {last_history["x_opt"]}\nFuction(x_opt) = {last_history["f_opt"]}")

        # print iteration
        headers = ["Iter", "lr", "x", "", "f(x)", "|f(x*) - f(x)|"]
        print(
            f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12}"
        )
        print("-" * 75)
        for iter in self._history:
            print(f"{iter["iteration"]:<6} "
                  f"{iter["lr"]:<12.4f} "
                  f"{iter["x_opt"][0]:<12.6f} "
                  f"{iter["x_opt"][1]:<12.6f} "
                  f"{iter["f_opt"]:<12.6f} "
                  f"{abs(last_history["f_opt"] - iter["f_opt"]):<12.6f}")
        print("=" * 75)

    def _numerical_derivative(self, x: Union[list, np.ndarray], order: int = 1) -> np.ndarray:
        pass
