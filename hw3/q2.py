import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import MultiNormal

np.random.seed(42)

DATA_PATH = os.path.join('data', 'trivariatenormal.dat')
data = pd.read_table(DATA_PATH, sep=' ')

OUTPUT_DIR = os.path.join("figure")


def main():
    mn = MultiNormal(data=data, max_iter=30, eps=1e-4, no_print=False)
    mn.optim()
    mn.plot(os.path.join(OUTPUT_DIR, "MultiNormal_EM_Plot.png"))


if __name__ == "__main__":
    main()
