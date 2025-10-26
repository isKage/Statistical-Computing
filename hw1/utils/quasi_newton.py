import numpy as np
from typing import Union, List, Optional


class MultivariateQuasiNewtonOptimizer:
    """Rank-One Quasi-Newton Optimizer"""

    def __init__(self, f, grad_f, lr: float = 0.01, epsilon: float = 1e-6, max_iter: int = 100, step_halving: bool = True):
        """Initialize Quasi-Newton Optimizer

        Args:
            f (_type_): objective function
            grad_f (_type_): gradient function, returns np.array shape (p,)
            lr (float, optional): learning rate. Defaults to 0.01.
            epsilon (float, optional): precision. Defaults to 1e-6.
            max_iter (int, optional): max iteration rounds. Defaults to 100.
            step_halving (bool, optional): whether to use step-halving. Defaults to True.
        """
        self.f = f
        self.grad_f = grad_f
        self.lr = lr
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.step_halving = step_halving

        self._start_point = None
        self._history = []

    def clear(self):
        """clear history"""
        self._history = []

    def optimize(self, x0: Union[List[float], np.ndarray], M: Optional[np.ndarray] = None) -> dict:
        """begin Quasi-Newton optimization

        Args:
            x0 (Union[List[float], np.ndarray]): start point
            M (Optional[np.ndarray], optional): initial weight matrix. Defaults to None, which means identity matrix.

        Returns:
            dict: result
        """
        self._start_point = x0
        x = np.array(x0, dtype=float)

        self._start_point = x.copy()
        self._history = [
            {'iteration': 0, 'x_opt': x.copy(), 'f_opt': self.f(x), 'lr': self.lr}
        ]

        if M is None:
            # weight matrix (firstly identity matrix = diag(1, 1, ..., 1))
            M = - np.eye(len(x))
        # initial gradient
        grad = self.grad_f(x)

        # iteration
        for i in range(self.max_iter):
            # learning rate
            lr = self.lr
            # check M^{-1} exists or not
            if np.linalg.cond(M) > 1e12:
                # |M| |M^{-1}| > 1e12, ill-conditioned
                print(
                    f"Weight matrix (M) cannot be inversed at iter {i}. Early stop!"
                )
                break

            # step for update
            step = np.linalg.solve(M, grad)  # M @ step = grad

            # step-halving
            if self.step_halving:
                while self.f(x - lr * step) < self.f(x):
                    lr = lr / 2
                    if lr < 1e-6:
                        break

            # x update
            x_new = x - lr * step
            grad_new = self.grad_f(x_new)

            # convergence check, lr * step = x_new - x
            if (np.linalg.norm(lr * step) / (np.linalg.norm(x) + self.epsilon)) < self.epsilon:
                x = x_new
                grad = grad_new
                self._history.append(
                    {'iteration': i + 1, 'x_opt': x.copy(), 'f_opt': self.f(x),
                     'lr': lr}
                )
                break

            # weight matrix update: rank-one (important)
            z = (x_new - x).reshape(-1, 1)  # transpose to column vector
            y = (grad_new - grad).reshape(-1, 1)  # transpose to column vector
            v = y - M @ z
            c = v.T @ z
            if abs(c) > 1e-12:  # division should not be too small like 0
                c = 1 / c
                M = M + c * (v @ v.T)

            # go on
            x = x_new
            grad = grad_new
            self._history.append(
                {'iteration': i + 1, 'x_opt': x.copy(), 'f_opt': self.f(x), 'lr': lr}
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
            f"Coeffients:\nstart point = {self._start_point}\ninitial learning rate = {self.lr}\nepsilon = {self.epsilon}\nmax iteration = {self.max_iter}\nstep halving = {self.step_halving}"
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
