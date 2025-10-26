import os
import math
import numpy as np

from utils import BisectionMethod


def f(x):
    """objective function"""
    if x is None:
        return None  # in case of x = None
    return x**2 - 10


def main():
    # bisection method
    bisection_method = BisectionMethod(
        f, a=0.0, b=10.0, epsilon=1e-6, max_iter=50
    )
    root = bisection_method.solve()
    bisection_method.summary()

    print(f"Actual root = {math.sqrt(10)}")


if __name__ == "__main__":
    main()

    """
    python3 q1.py
    """
