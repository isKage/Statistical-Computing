import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import GeneticMapping, RandomCombinatorial, ParallelRandomCombin, ParallelSolver

# random seed
np.random.seed(42)

DATA_PATH = os.path.join("data", "gene.csv")
P = 12
N_INIT_POINTS = 15
MAX_ITERATION = 200
NEIGHBORHOOD = 20


def run_single():
    """single initial points"""
    # initial points
    theat_init = np.random.permutation(P)

    # initialize RandomCombinatorial
    rc = RandomCombinatorial(
        data_file_path=DATA_PATH,
        max_iteration=MAX_ITERATION,
        no_print=False
    )

    # optimizing
    rc.optim(theta_init=theat_init, neighborhood_size=NEIGHBORHOOD)

    # plot
    figure_file_path = os.path.join("figure", "random_combin.png")
    rc.plot(figure_file_path=figure_file_path)


"""exchange to a better multi parallel"""
# def run_multiple():
#     """multiple initial points, parallel running"""
#     # initial points
#     theta_inits = [np.random.permutation(P) for i in range(N_INIT_POINTS)]

#     # initialize ParallelRandomCombin
#     prc = ParallelRandomCombin(
#         data_file_path=DATA_PATH,
#         max_iteration=100,
#         works=None,  # cpu core num: 8
#         neighborhood_size=20
#     )

#     # optimizing
#     prc.optim(theta_inits=theta_inits)

#     # plot
#     figure_file_path = os.path.join("figure", "multi_random_combin.png")
#     prc.plot(figure_file_path=figure_file_path)


def run_multiple():
    """multiple initial points, parallel running"""
    # initial points
    theta_inits = [np.random.permutation(P) for i in range(N_INIT_POINTS)]

    # task list
    task_list = [
        {
            "solver": RandomCombinatorial,
            "init_kwargs": {
                "data_file_path": DATA_PATH,
                "max_iteration": MAX_ITERATION,
                "no_print": True
            },
            "optim_kwargs": {
                "theta_init": theat_init,
                "neighborhood_size": NEIGHBORHOOD
            }
        } for theat_init in theta_inits
    ]

    # initialize ParallelSolver
    ps = ParallelSolver(works=os.cpu_count(), check="theta_init")

    # optimizing
    ps.optim(task_list=task_list)

    # plot
    figure_file_path = os.path.join("figure", "multi_random_combin.png")
    ps.plot(figure_file_path=figure_file_path)


def main():
    run_single()
    print("\n\n\n")
    run_multiple()


if __name__ == "__main__":
    main()
