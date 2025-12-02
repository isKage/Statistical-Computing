import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import SIS, SISParticleFilter

np.random.seed(42)

# observed data
y = np.array([
    2, 1, 6, 2, 1, 6, 6, 5, 6, 6, 6, 3, 5, 2, 3, 2, 1,
    2, 6, 4, 6, 2, 2, 5, 3, 3, 3, 1, 4, 3, 1, 5, 1, 3,
    6, 1, 6, 3, 5, 1, 6, 3, 1, 2, 3, 1, 4, 6, 3, 6, 5,
    1, 3, 3, 5, 6, 1, 3, 5, 5, 4, 6, 3, 2, 4, 1, 6, 2,
    5, 4, 2, 4, 4, 2, 1, 2, 3, 2, 6, 3, 6, 6, 6, 4, 5,
    6, 2, 2, 4, 6, 6, 1, 4, 6, 3, 4, 2, 6, 4, 6
], dtype=int)

# real hidden data
z_star = np.array([
    1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0
], dtype=int)

# sample num
N = 10000
# length of seq
T = 100
# Transition matrix
Tmat = np.array([
    [0.9, 0.1],   # z_{t-1} = 0
    [0.05, 0.95]  # z_{t-1} = 1
])


def SIS_no_PF():
    """SIS: Sequential Importance Sampling"""
    sis = SIS(N, T, Tmat)
    sis.SIS(y)
    sis.summary(z_star)


def SIS_PF():
    """SIS: Sequential Importance Sampling with Particle Filter"""
    sispf = SISParticleFilter(N, T, Tmat, N / 7)
    sispf.SIS(y)
    sispf.summary(z_star)
    sispf.plot_ess(os.path.join("figure", "ess_plot.png"))


if __name__ == "__main__":
    print("# SIS: Sequential Importance Sampling #")
    SIS_no_PF()
    print()
    print("# SIS: Sequential Importance Sampling with Particle Filter #")
    SIS_PF()
