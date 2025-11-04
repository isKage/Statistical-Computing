import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import SimulatedAnnealing, ParallelSolver

# random seed
np.random.seed(42)

DATA_PATH = os.path.join("data", "gene.csv")
P = 12
MAX_ITERATION = 1000

TEMPERATURE_INIT = 10
STAGE_INIT = 100
NEIGHBORHOOD_SIZE = 2


def cooling_func(t):
    """cooling function"""
    return 0.9 * t


def stage_func(m):
    """stage length function"""
    return m + 40


def run_single():
    """single initial points"""
    # initial points
    theat_init = np.random.permutation(P)

    # initialize SimulatedAnnealing
    sa = SimulatedAnnealing(
        data_file_path=DATA_PATH,
        max_iteration=MAX_ITERATION,
        neighborhood_size=NEIGHBORHOOD_SIZE,
        no_print=False
    )

    # optimizing
    sa.optim(
        cooling_func=cooling_func,
        stage_func=stage_func,
        theta_init=theat_init,
        temperature_init=TEMPERATURE_INIT,
        stage_init=STAGE_INIT
    )

    # plot
    figure_file_path = os.path.join("figure", "simulated_annealing_combin.png")
    sa.plot(figure_file_path=figure_file_path)


def run_multiple():
    """multiple initial points, parallel running"""
    # initial points
    theat_init = np.random.permutation(P)

    # task list
    temperature_init_list = [1, 5, 10, 15, 50, 100]
    task_list = [
        {
            "solver": SimulatedAnnealing,
            "init_kwargs": {
                "data_file_path": DATA_PATH,
                "max_iteration": MAX_ITERATION,
                "neighborhood_size": NEIGHBORHOOD_SIZE,
                "no_print": True
            },
            "optim_kwargs": {
                "cooling_func": cooling_func,
                "stage_func": stage_func,
                "theta_init": theat_init,
                "temperature_init": ti,
                "stage_init": STAGE_INIT
            }
        } for ti in temperature_init_list
    ]

    # initialize ParallelSolver
    ps = ParallelSolver(works=os.cpu_count(), check="temperature_init")

    # optimizing
    ps.optim(task_list=task_list)

    # plot
    figure_file_path = os.path.join(
        "figure", "multi_simulated_annealing_combin.png")
    ps.plot(figure_file_path=figure_file_path)


def main():
    run_single()
    print("\n\n\n")
    run_multiple()


if __name__ == "__main__":
    main()
