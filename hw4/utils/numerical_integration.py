import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict, Optional, Union

LOG_NUM = 80


class NewtonCotesQuadrature:
    """Numerical Integration: Newton Cotes Quadrature"""

    def __init__(self, f: callable, a: float, b: float, n: int = 10, rule: str = "Riemann"):
        """Initialize Newton Cotes Quadrature Method for Numerical Integration

        Args:
            f (callable): function for integration
            a (float): low, integrate: [a, b]
            b (float): high, integrate: [a, b]
            n (int, optional): number of intervals. Defaults to 10.
            rule (str, optional): Quadrature Rule. Defaults to "Riemann".
        """
        self.f = f
        self.a = a
        self.b = b
        self.n = int(n)
        self.rule = rule

        # length of a tiny intervals
        self.h = float(abs(b-a) / n)
        # n + 1 points: a = x0 < x1 < ... < xn = b
        self.xi = np.linspace(a, b, n + 1)

        # final result
        self.integration = None

    def integrate(self) -> float:
        """Integrate"""
        if self.rule == "Riemann":
            self.integration = self._riemann_rule()
        elif self.rule == "Trapezoidal":
            self.integration = self._trapezoidal_rule()
        elif self.rule == "Simpson":
            self.integration = self._simpson_rule()
        else:
            raise ValueError(
                "Choose one rule from \"Riemann\", \"Trapezoidal\" or \"Simpson\"!"
            )
        return self.integration

    def integrate_compare(self, real=None):
        """Show the comparision among these 3 rules (IMPORTANT: n is fixed, that is, number of intervals is fixed!)

        Args:
            real (float, optional): Actual value of integration. Defaults to None.
        """
        n_map = {0: "Riemann", 1: "Trapezoidal", 2: "Simpson"}
        # calculate int applying these 3 rule
        riemann_int = self._riemann_rule()
        trapezoidal_int = self._trapezoidal_rule()
        simpson_int = self._simpson_rule()
        ints = [riemann_int, trapezoidal_int, simpson_int]

        # calculate the error if given `real`
        if real is not None:
            e_r = abs(riemann_int - real)
            e_t = abs(trapezoidal_int - real)
            e_s = abs(simpson_int - real)
            es = [e_r, e_t, e_s]

        # print
        print("=" * LOG_NUM)
        print(f"# Actual Int: F = {real:.10f} @ [{self.a:.2f}, {self.b:.2f}]")
        print("-" * LOG_NUM)
        if real is not None:
            print("Rule\t\tF\t\tError")
            for i in range(3):
                print(f"{n_map[i]:<12}\t{ints[i]:.10f}\t{es[i]:.10f}")
        else:
            print("Rule\t\tF")
            for i in range(3):
                print(f"{n_map[i]:<12}\t{ints[i]:.10f}")

        # conclusion
        print("-" * LOG_NUM)
        imin = np.argmin(np.array(es))
        if real is not None:
            print(
                f"{n_map[imin]} best, with F = {ints[imin]:.10f}, error = {es[imin]:.10f}"
            )
        else:
            print(f"{n_map[imin]} best, with F = {ints[imin]:.10f}")
        print("=" * LOG_NUM)

    def _riemann_rule(self) -> float:
        """Riemann Rule"""
        try:
            fvals = self.f(self.xi[:-1])
            # sum h * f(xi)
            return np.sum(self.h * fvals)
        except:
            raise ValueError("Error occurs with Riemann Rule!")

    def _trapezoidal_rule(self) -> float:
        """Trapezoidal Rule"""
        try:
            fvals = self.f(self.xi)
            # [f(a)/2 + f(b)/2] * h + sum h * f(xi)
            return self.h * (0.5 * fvals[0] + np.sum(fvals[1:-1]) + 0.5 * fvals[-1])
        except:
            raise ValueError("Error occurs with Trapezoidal Rule!")

    def _simpson_rule(self) -> float:
        """Simpson Rule"""
        try:
            fvals = self.f(self.xi)
            # h/3 sum [ f(x_{2i-2}) + 4 f(x_{2i-1}) + f(x_{2i}) ]
            return (self.h / 3) * (
                fvals[0]
                + fvals[-1]
                + 4 * np.sum(fvals[1:-1:2])
                + 2 * np.sum(fvals[2:-2:2])
            )
        except:
            raise ValueError("Error occurs with Trapezoidal Rule!")


class GaussianQuadrature:
    """Numerical Integration: Gaussian Quadrature"""

    def __init__(self, f: callable, a: float = None, b: float = None, n: int = 10):
        """Initialize Gaussian Quadrature Method for Numerical Integration

        Args:
            f (callable): function for integration
            a (float, optional): low, integrate: [a, b]
            b (float, optional): high, integrate: [a, b]
            n (int, optional): number of intervals. Defaults to 10.
        """
        self.f = f
        self.a = a
        self.b = b
        self.n = n

        # final result
        self.integration = None

    def legendre(self, xi=None, Ai=None) -> float:
        """Gauss–Legendre, (-1, 1)"""
        # check a and b
        if self.a is None or self.b is None:
            raise ValueError("Gauss–Legendre requires finite interval [a, b]")

        # a < x0 < ... < xn < b, and Ai
        if xi is not None and Ai is not None:
            xi, Ai = np.array(xi), np.array(Ai)
        else:
            xi, Ai = np.polynomial.legendre.leggauss(self.n)

        # xi (-1, 1) -> (a, b)
        yi = (self.b - self.a) / 2 * xi + (self.a + self.b) / 2
        # jacobi: dxi /dyi = (b - a) / 2
        self.integration = (self.b - self.a) / 2 * np.sum(Ai * self.f(yi))

        return self.integration

    def summary(self, real=None):
        """Not important"""
        if real is not None:
            error = abs(self.integration - real)
        print("=" * LOG_NUM)
        if real is not None:
            print(
                f"# Actual Int: F = {real:.10f} @ [{self.a:.2f}, {self.b:.2f}]")
            print("-" * LOG_NUM)

        if real is not None:
            print("Method\t\tF\t\tError")
            print(f"Gauss–Legendre\t{self.integration:.10f}\t{error:.10f}")
        else:
            print("Method\t\tF")
            print(f"Gauss–Legendre\t{self.integration:.10f}")

        print("-" * LOG_NUM)
        if real is not None:
            print(
                f"# Gauss–Legendre: F = {self.integration:.10f}, error = {error:.10f}"
            )
        else:
            print(f"# Gauss–Legendre: F = {self.integration:.10f}")
        print("=" * LOG_NUM)
