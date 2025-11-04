import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import GeneticAlgorithm, ParallelSolver

# random seed
np.random.seed(42)

# parameters
DATA_PATH = os.path.join("data", "gene.csv")
P = 12
MAX_ITERATION = 500

NUM_ORGANISMS = 40
CODING_METHOD = "order"
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.01
REPLACE_RATE = 0.3
REPLACE_RULE = "parents"

NUM_CROSSOVER = 2
ELITE_RATE = 0.1
K = 8


def replace_rate_func(replace_rate: float, iter: int | None = None) -> float:
    """you can define your replace rate function here, or just use fixed replace rate"""
    return replace_rate


def run_single():
    """single initial points"""
    # initial points
    theat_init = np.random.permutation(P)

    # initialize GeneticAlgorithm
    ga = GeneticAlgorithm(
        data_file_path=DATA_PATH,
        num_organisms=NUM_ORGANISMS,
        coding_method=CODING_METHOD,
        crossover_rate=CROSSOVER_RATE,
        mutation_rate=MUTATION_RATE,
        replace_rate=REPLACE_RATE,  # initial replace rate, defaults to 1/p
        replace_rule=REPLACE_RULE,
        max_iteration=MAX_ITERATION,
        no_print=False,
        debug=False  # debug test
    )

    # optimizing
    ga.optim(
        replace_rate_func=replace_rate_func,
        thetas=None,  # defaults to random generation
        elite_rate=ELITE_RATE,  # if add the current best back to population
        num_crossover=NUM_CROSSOVER,  # num of crossover points
        swap=1,  # for order coding, we need set every crossover swapping times
        k_tournament=K
    )

    # plot
    figure_file_path = os.path.join("figure", "genetic_algorithm_combin.png")
    ga.plot(figure_file_path=figure_file_path, plot_type="line")


def main():
    run_single()


if __name__ == "__main__":
    main()
