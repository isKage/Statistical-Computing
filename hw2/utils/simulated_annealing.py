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


class SimulatedAnnealing(GeneticMapping, BasicSolver):
    """Simulated Annealing"""

    def __init__(self, data_file_path: str, max_iteration: int = 100, neighborhood_size: int = 20, no_print: bool = False):
        """initialize

        Args:
            data_file_path (str): data file, for loading
            max_iteration (int, optional): max iter rounds. Defaults to 100.
            neighborhood_size (int, optional): neighborhood size of every theta. Defaults to 20.
            no_print (bool, optional): print some info or not. Defaults to False.
        """
        # initialize the parent class
        GeneticMapping.__init__(self, data_file_path)
        BasicSolver.__init__(self, max_iteration)

        # some param
        self.neighborhood_size = neighborhood_size

        self.no_print = no_print
        if not self.no_print:
            print("="*LOG_N_LINE)
            print("Simulated Annealing Method")
            print(f"Sample data from : {data_file_path}")
            print(f"Max Iteration: {max_iteration}")

    def neighborhood(self, theta: np.ndarray) -> List[np.ndarray]:
        """Generate neighborhood by swapping locs

        Args:
            theta (np.ndarray): current solution

        Returns:
            List[np.ndarray]: list of candidate solutions
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
            neighborhood.append(new_theta)
        return neighborhood  # [theta1, theta2, ...]

    def optim(self, cooling_func: callable, stage_func: callable, theta_init: Optional[list | np.ndarray] = None, temperature_init: float = 5, stage_init: int = 50) -> dict:
        """optim of simulated annealing

        Args:
            cooling_func (callable): function for temperature cooling, annealing
            stage_func (callable): function for stage length changing
            theta_init (Optional[list  |  np.ndarray], optional): initial point. Defaults to None.
            temperature_init (float, optional): initial temperature. Defaults to 5.
            stage_init (int, optional): initial stage length. Defaults to 50.

        Returns:
            dict: result
        """
        # valid solution
        theta = self._valid_solution(theta=theta_init)

        # print some info
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print(
                f"Initial Solution: {theta_init}\tInitial Temperature: {temperature_init}\nInitial Stage length: {stage_init}"
            )
            print("Updating Process:\n")

        # record theta0
        f_val = self.log_likelihood(theta)
        self.record(0, theta, f_val)

        # temperature: t and stage length: m
        t = temperature_init
        m = stage_init
        current_stage_length = 0  # init stage length

        for iter in range(1, self.max_iteration + 1):
            # 1. update stage
            current_stage_length += 1

            # 2. calculate the candidate x and f
            neighbors = self.neighborhood(theta)
            # save the best one
            theta_new = max(neighbors, key=self.log_likelihood)
            f_new = self.log_likelihood(theta_new)

            # 3. update theta to new or not
            delta = f_new - f_val
            prob = min(1, np.exp(delta / t))
            # if (delta > 0) or (np.log(np.random.rand()) < delta / t):  # U(0, 1)
            if np.random.rand() < prob:  # U(0, 1)
                theta = theta_new
                f_val = f_new

            # 4. check current stage and then annealing
            if current_stage_length == m:
                m = stage_func(m)
                t = cooling_func(t)
                current_stage_length = 0

            # record in history
            self.record(iteration=iter, theta=theta, f_value=f_val)

            if (not self.no_print) and (iter % max(1, self.max_iteration // 10) == 0):
                print(
                    f"Iter {iter}\tTemper: {t:.4f}\tCurrent Func = {f_val:.4f}\tOpt Func = {self.f_opt:.4f}"
                )

        # print end info
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print(f"Opt Solution: {list(self.theta_opt)}")
            print(f"Opt Objective Func: {self.f_opt}")
            print(f"Initial Temperature: {temperature_init}")
            print(f"Initial Stage length: {stage_init}")
            print("="*LOG_N_LINE)

        return {
            "theta_opt": self.theta_opt,
            "f_opt": self.f_opt,
            "history": self.history,
            "theta_init": theta_init,
            "temperature_init": temperature_init,
            "stage_init": stage_init
        }
