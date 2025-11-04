import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import RandomCombinatorial, TabuSolver, ParallelSolver

# random seed
np.random.seed(42)

DATA_PATH = os.path.join("data", "gene.csv")
P = 12
N_INIT_POINTS = 15
MAX_ITERATION = 200


def run_single():
    """single initial points"""
    # initial points
    theat_init = np.random.permutation(P)

    # initialize TabuSolver
    ts = TabuSolver(
        data_file_path=DATA_PATH,
        max_iteration=MAX_ITERATION,
        neighborhood_size=20,
        tabu_tenure=5,
        no_print=False
    )

    # optimizing
    ts.optim(theta_init=theat_init)

    # plot
    figure_file_path = os.path.join("figure", "tabu_combin.png")
    ts.plot(figure_file_path=figure_file_path)


def run_multiple():
    """multiple initial points, parallel running"""
    # initial points
    theat_init = np.random.permutation(P)

    # task list
    tabu_tenure_list = [3, 5, 7, 9, 11, 13]
    task_list = [
        {
            "solver": TabuSolver,
            "init_kwargs": {
                "data_file_path": DATA_PATH,
                "max_iteration": MAX_ITERATION,
                "neighborhood_size": 20,
                "tabu_tenure": tt,
                "no_print": True
            },
            "optim_kwargs": {
                "theta_init": theat_init
            }
        } for tt in tabu_tenure_list
    ]

    # initialize ParallelSolver
    ps = ParallelSolver(works=os.cpu_count(), check="tabu_tenure")

    # optimizing
    ps.optim(task_list=task_list)

    # plot
    figure_file_path = os.path.join("figure", "multi_tabu_combin.png")
    ps.plot(figure_file_path=figure_file_path)


def main():
    run_single()
    print("\n\n\n")
    run_multiple()


if __name__ == "__main__":
    main()
