import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import NewtonCotesQuadrature, GaussianQuadrature

REAL = 2/3
XI = np.array([
    0.148874338981631,
    0.433395394129247,
    0.679409568299024,
    0.865063366688985,
    0.973906528517172
])
XI = np.concatenate((-XI[::-1], XI))
AI = np.array([
    0.295524224714753,
    0.269266719309996,
    0.219086362515982,
    0.149451394150581,
    0.066671344308688
])
AI = np.concatenate((AI[::-1], AI))


def f(x):
    """Target Function"""
    # int f = x**3 / 3
    return x**2


def newton():
    ncq = NewtonCotesQuadrature(f=f, a=-1, b=1, n=10)
    ncq.integrate_compare(real=REAL)


def gaussian():
    gq = GaussianQuadrature(f=f, a=-1, b=1, n=10)
    # res = gq.legendre(xi=XI, Ai=AI)
    res = gq.legendre()  # not use XI, AI
    gq.summary(real=REAL)


if __name__ == "__main__":
    newton()
    print("\n\n")
    gaussian()
