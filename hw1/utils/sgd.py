import numpy as np
from abc import ABC, abstractmethod  # abstract class
import matplotlib.pyplot as plt


class BaseSGD(ABC):
    """Base class of SGD optimizer, an abstract class"""

    def __init__(self, f, grad_f, lr: float = 0.01, batch_size: int = 1, epsilon: float = 1e-6, max_iter: int = 100, repeat: bool = False):
        """
        Base class for SGD optimizer

        Args:
            f (function): objective function
            grad_f (function): gradient function, return np.ndarray (p,)
            lr (float): learning rate, default to 0.01.
            batch_size (int): batch size, default to 1.
            epsilon (float): precision, default to 1e-6.
            max_iter (int): max iteration count, default to 100.
            repeat (bool): sampling with replacement or not, default to False.
        """
        self.f = f
        self.grad_f = grad_f
        self.lr = lr
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.repeat = repeat

        self._start_point = None
        self._history = []
        self._result = None
        self._epoch = None  # final iteration * batch_size / total sample number

        # save the final answer
        self._x_opt = None
        self._f_opt = None
        self._iter_opt = None

    @abstractmethod
    def step(self, x: np.ndarray, grad: np.ndarray):
        """update x (abstract method, waiting for create)

        Args:
            x (np.ndarray): x_t
            grad (np.ndarray): gradient at x_t
        """
        raise NotImplementedError

    def sampling(self, n_samples: int):
        """sampling from samples

        Args:
            n_samples (int): total number of samples

        Raises:
            ValueError: while without replacement, batch size < total sample number

        Returns:
            np.ndarray: index of sample from N samples
        """
        if self.repeat:
            # with replacement, meaning existing repeated index
            return np.random.randint(0, n_samples, size=self.batch_size)
        else:
            # without replacement, choose batch size sample from all samples
            if self.batch_size >= n_samples:  # > or >= not very different
                raise ValueError(
                    "Batch size larger than total sample size without replacement.")
            return np.random.choice(n_samples, size=self.batch_size, replace=False)

    def optimize(self, x0: np.ndarray, n_samples: int) -> dict:
        """begin optimize

        Args:
            x0 (np.ndarray): start point
            n_samples (int): total sample number

        Returns:
            dict: result
        """
        # initial start point
        self._start_point = np.array(x0, dtype=float)
        x = self._start_point.copy()

        # first history
        self._history = [{
            "iteration": 0,
            "lr": self.lr,
            "x_opt": x.copy(),
            "f_opt": self.f(x)
        }]

        # begin iter
        for i in range(self.max_iter):
            # sampling
            batch_idx = self.sampling(n_samples)
            # batch gradient
            grad = self.grad_f(x, batch_idx)  # Important!!!

            # update
            x_new = self.step(x, grad)

            # convergence check
            if np.linalg.norm(x_new - x) / (np.linalg.norm(x) + self.epsilon) < self.epsilon:
                break

            # go on
            x = x_new

            self._history.append({
                "iteration": i + 1,
                "lr": self.lr,
                "x_opt": x.copy(),
                "f_opt": self.f(x)
            })

        # record the epoch
        self._epoch = int((i + 1) * self.batch_size / n_samples)

        if self._history:
            opt = max(self._history, key=lambda h: h["f_opt"])
            self._x_opt = opt["x_opt"].copy()
            self._f_opt = opt["f_opt"]
            self._iter_opt = opt["iteration"]

        # result
        return {
            "iteration": i + 1,
            "lr": self.lr,
            "x_opt": x.copy(),
            "f_opt": self.f(x)
        }

    def plot(self, figure_file_path: str | None = None, title: str | None = None, show: bool = False):
        """plot objective function f(x)

        Args:
            figure_file_path (str | None, optional): file path, for saving. Defaults to None.
            show (bool, optional): show plot or not. Defaults to False.
        """
        if len(self._history) == 0:
            print("Must apply optimize() function first.")
            return None

        f = [iter["f_opt"] for iter in self._history]

        plt.figure(figsize=(10, 6))
        plt.plot(f, marker='o', linewidth=0.8, markersize=3)
        if title:
            plt.title(f"Objective Function {title}")
        else:
            plt.title("Objective Function with iteration")
        plt.xlabel("Iteration")
        plt.ylabel("f(x)")
        plt.grid(True)

        # Save figure
        if figure_file_path:
            plt.savefig(figure_file_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {figure_file_path}")

        if show:
            plt.show()

    def summary(self):
        """summary"""
        if len(self._history) == 0:
            print("Must apply optimize() function first.")
            return None

        print("=" * 60)
        print(
            f"Coeffients:\nstart point = {self._start_point}\ninitial learning rate = {self.lr}\nepsilon = {self.epsilon}\nmax iteration = {self.max_iter}\nrepeat = {self.repeat}\nEpoch = {self._epoch}"
        )
        print("-" * 60)
        last_history = self._history[-1]
        print(
            f"x_opt = {self._x_opt}, Function(x_opt) = {self._f_opt}, Iteration = {self._iter_opt}"
        )
        print(
            f"x_last = {last_history['x_opt']}, Function(x_last) = {last_history['f_opt']:.6f}, Iteration = {last_history['iteration']}")
        print("-" * 60)

        headers = ["Iter", "x", "", "f(x)", "|f(x*) - f(x)|"]
        print(
            f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12}"
        )
        print("-" * 60)
        for i, iter in enumerate(self._history):
            if i % 10 == 0:
                print(f"{iter["iteration"]:<6} "
                      f"{iter["x_opt"][0]:<12.6f} "
                      f"{iter["x_opt"][1]:<12.6f} "
                      f"{iter["f_opt"]:<12.6f} "
                      f"{abs(last_history["f_opt"] - iter["f_opt"]):<12.6f}")
        print("=" * 60)

    def history(self):
        """return history"""
        return self._history


class BatchSGD(BaseSGD):
    """Mini Batch SGD"""

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """create step(), SGD update

        Args:
            x (np.ndarray): x
            grad (np.ndarray): gradient

        Returns:
            np.ndarray: x_new
        """
        return x + self.lr * grad  # uphill (not important, just + or -)


class MomentumSGD(BaseSGD):
    """Momentum SGD"""

    def __init__(self, f, grad_f, lr: float = 0.01, batch_size: int = 1, beta: float = 0.01, epsilon: float = 1e-6, max_iter: int = 100, repeat: bool = False):
        """initial Momentum SGD

        Args:
            beta (float, optional): step of Momentum. Defaults to 0.01.
            others are the same as BaseSGD
        """
        # BaseSGD initial
        super().__init__(f, grad_f, lr, batch_size, epsilon, max_iter, repeat)

        self.beta = beta

        self.delta = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """create step(), Momentum SGD update

        Args:
            x (np.ndarray): x
            grad (np.ndarray): gradient

        Returns:
            np.ndarray: x_new
        """
        if self.delta is None:
            # initial as (0, 0, ..., 0) in R^p, same as x
            self.delta = np.zeros_like(x)

        # update delta_t
        self.delta = self.beta * self.delta + grad

        return x + self.lr * self.delta  # uphill, update x


class RMSPropMomentumSGD(BaseSGD):
    """RMSProp Momentum SGD"""

    def __init__(self, f, grad_f, lr: float = 0.01, batch_size: int = 1, beta: float = 0.01, rho: float = 0.01, epsilon: float = 1e-6, max_iter: int = 100, repeat: bool = False):
        """Initial RMSProp Momentum SGD

        Args:
            beta (float, optional): step of Momentum. Defaults to 0.01.
            rho (float, optional): weight. Defaults to 0.01.
            others are the same as BaseSGD.
        """
        # BaseSGD initial
        super().__init__(f, grad_f, lr, batch_size, epsilon, max_iter, repeat)

        self.beta = beta
        self.rho = rho

        self.delta = None
        self.G = None

    def step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """create step(), RMSProp Momentum SGD update

        Args:
            x (np.ndarray): x
            grad (np.ndarray): gradient

        Returns:
            np.ndarray: x_new
        """
        # initial as (0, 0, ..., 0) in R^p, same as x
        if self.delta is None:
            self.delta = np.zeros_like(x)

        # initial as (0, 0, ..., 0) in R^p, same as x
        if self.G is None:
            self.G = np.zeros_like(x)

        # update G_t
        self.G = self.rho * self.G + (1 - self.rho) * grad ** 2

        # update delta_t, calculate every element, not matrix calculate
        self.delta = self.beta * self.delta + \
            grad / (np.sqrt(self.G) + self.epsilon)

        return x + self.lr * self.delta  # uphill
