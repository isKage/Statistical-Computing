import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .basic import BasicSolver
from .genetic_mapping import GeneticMapping

LOG_N_LINE = 100


class TabuSolver(GeneticMapping, BasicSolver):
    """Tabu Solver"""

    def __init__(self, data_file_path: str, max_iteration: int = 100, neighborhood_size: int = 20, tabu_tenure: int = 5, no_print: bool = False):
        """initialize tabu

        Args:
            data_file_path (str): data file path, for load data
            max_iteration (int, optional): max iteration rounds. Defaults to 100.
            neighborhood_size (int, optional): neighborhood size. Defaults to 20.
            tabu_tenure (int, optional): _description_. timing for a tabu action to 5.
            no_print (bool): print or not. Defaults to False.
        """
        # initialize the parent class
        GeneticMapping.__init__(self, data_file_path)
        BasicSolver.__init__(self, max_iteration)

        self.tabu_tenure = tabu_tenure
        self.neighborhood_size = neighborhood_size  # size of every neighbors
        self.no_print = no_print  # print some info or not
        if not self.no_print:
            print("="*LOG_N_LINE)
            print("Tabu Method")
            print(f"Sample data from : {data_file_path}")
            print(f"Max Iteration: {max_iteration}")

    def neighborhood(self, theta: np.ndarray) -> List[Tuple[Tuple[int, int], np.ndarray]]:
        """Generate neighborhood by swapping locs

        Args:
            theta (np.ndarray): current solution

        Returns:
            List[Tuple[Tuple[int, int], np.ndarray]]: [((i, j), [1 2 3]), ...]
        """
        # all idx pairs
        p = self.p
        max_pairs = p * (p - 1) // 2  # actual num
        # sample size <= min(set num, actual num)
        sample_size = min(self.neighborhood_size, max_pairs)
        # all idx pairs, i < j
        all_idx_pairs = [(i, j) for i in range(p) for j in range(i+1, p)]

        # select sample_size as samples
        idx_sampled = np.random.choice(
            len(all_idx_pairs), size=sample_size, replace=False
        )
        pairs = [all_idx_pairs[k] for k in idx_sampled]

        # generate neighborhood
        neighborhood = []
        for (i, j) in pairs:
            # as for the generation method of (i, j), we have (i, j)
            new_theta = theta.copy()
            new_theta[i], new_theta[j] = new_theta[j], new_theta[i]  # swap
            neighborhood.append(((i, j), new_theta))
        return neighborhood  # with (i, j) as id

    def optim(self, theta_init: Optional[list | np.ndarray] = None) -> dict:
        """begin optim

        Args:
            theta_init (Optional[list  |  np.ndarray], optional): initial point. Defaults to None.

        Returns:
            dict: results
        """
        # valid solution
        theta = self._valid_solution(theta=theta_init)

        # print some info
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print(
                f"Initial Solution: {theta_init}\nTabu Tenure: {self.tabu_tenure}"
            )
            print("Updating Process:\n")

        f_val = self.log_likelihood(theta)
        self.record(0, theta, f_val)

        # tabu dict: {(i,j): timing}, swap (i, j) is not allowed unless timing  = 0. (note: i < j)
        tabu_dict = {}

        for iter in range(1, self.max_iteration + 1):
            # 1. generate neighbors
            neighbors = self.neighborhood(theta)

            # record the state of this iter
            current_action_opt = None
            current_theta_opt = None
            current_f_opt = -np.inf

            for ((i, j), theta_new) in neighbors:
                # for swapping (i, j)
                action = (min(i, j), max(i, j))
                f_new = self.log_likelihood(theta_new)  # new f value

                # 2. check if it is in tabu dict, and timing > 0
                is_tabu = action in tabu_dict and tabu_dict[action] > 0
                # 3. Aspiration: allowed if better than global best
                if is_tabu and f_new <= self.f_opt:
                    # not good enough and in tabu, so no action and next solution
                    continue

                # 4. action if not in tabu dict and better
                if f_new > current_f_opt:
                    current_action_opt = action  # record, maybe we use this action in this iter
                    current_theta_opt = theta_new
                    current_f_opt = f_new

            # 5. update: record current opt state
            theta = current_theta_opt
            f_val = current_f_opt

            # 6. update tabu dict: forbid the reverse action
            # swap two => (i, j) = (j, i), no different
            tabu_dict[current_action_opt] = self.tabu_tenure

            # 7. update tabu tenure of tabu dict
            expired = []  # less than 0
            for act in list(tabu_dict.keys()):
                tabu_dict[act] -= 1
                if tabu_dict[act] <= 0:
                    expired.append(act)
            # delete expired action
            for act in expired:
                del tabu_dict[act]

            # record
            self.record(iter, theta, f_val)

            # print some info
            if (not self.no_print) and (iter % max(1, self.max_iteration // 10) == 0):
                print(
                    f"Iter {iter}\tCurrent Func = {f_val:.4f}\tOpt Func = {self.f_opt:.4f}"
                )

        # print end info
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print(f"Opt Solution: {list(self.theta_opt)}")
            print(f"Opt Objective Func: {self.f_opt}")
            print(f"Tabu Tenure: {self.tabu_tenure}")
            print("="*LOG_N_LINE)

        return {
            "theta_opt": self.theta_opt,
            "f_opt": self.f_opt,
            "history": self.history,
            "theta_init": theta_init,
            "tabu_tenure": self.tabu_tenure
        }
