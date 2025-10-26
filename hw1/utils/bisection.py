import os
import math
import numpy as np


class BisectionMethod:
    """Bisection Method"""

    def __init__(self, f, a: float, b: float, epsilon: float = 1e-6, max_iter: int = 100):
        """Initial Bisection Method class

        Args:
            f (function): objective function
            a (float): left
            b (float): right
            epsilon (int): precision demand, Defaults to 1e-6.
            max_iter (int): max iteration number. Defaults to 100.
        """
        self.f = f
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.root = None

        self._iteration = 0  # actual iteration number
        self._history = []  # history root guess among iteration

        # check if root belong to [a, b]
        self._validate_interval()

    def solve(self) -> float:
        """Begin bisection method

        Returns:
            float: final root
        """
        a, b = self.a, self.b
        f = self.f

        self._history = []
        self._iteration = 0

        for i in range(self.max_iter):
            c = (a + b) / 2
            if self._iteration > 0:
                c_before = self._history[-1].get("root")
            fc = f(c)

            self._iteration += 1
            self._history.append({
                "iteration": self._iteration,
                "a": a,
                "b": b,
                "root": c,
                "f(root)": fc,
            })  # store the history info of every iteration

            # check if convergence
            if self._iteration > 1:
                if abs(fc) < self.epsilon or (abs(c - c_before) / (abs(c) + self.epsilon)) < self.epsilon:
                    # similar to root or |x(i+1) - x(i)| / [x(i) + epsilon] too small
                    self.root = c
                    return c

            # else, update the interval
            if f(a) * fc < 0:
                b = c
            else:
                a = c

        # iteration too much, choose the last (a + b) / 2
        self.root = (a + b) / 2
        return self.root

    def summary(self):
        """Summary of this iteration"""
        if len(self._history) == 0:
            print("Must apply solve() function first.")
            return None
        
        print("="*80)
        print(
            f"Coeffients:\na[0] = {self.a}, b[0] = {self.b}\nepsilon = {self.epsilon}\nmax iteration = {self.max_iter}"
        )
        print("-"*80)
        print(f"Root = {self.root}\nFuction(Root) = {self.f(self.root)}")

        # print iteration
        headers = ["Iter", "a", "b", "root", "f(root)", "len(Interval)"]
        print(
            f"{headers[0]:<6} {headers[1]:<12} {headers[2]:<12} {headers[3]:<12} {headers[4]:<12} {headers[5]:<12}"
        )
        print("-" * 80)
        for iter in self._history:
            print(f"{iter['iteration']:<6} "
                  f"{iter['a']:<12.6f} "
                  f"{iter['b']:<12.6f} "
                  f"{iter['root']:<12.6f} "
                  f"{iter['f(root)']:<+12.6f} "
                  f"{iter['b'] - iter['a']:<12.6f}")
        print("="*80)

    def _validate_interval(self):
        """check function"""
        if self.f(self.a) * self.f(self.b) > 0:
            raise ValueError(
                f"Interval [{self.a}, {self.b}] must satisfy f(a) * f(b) < 0")
