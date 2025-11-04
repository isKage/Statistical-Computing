import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .basic import BasicSolver
from .genetic_mapping import GeneticMapping

LOG_N_LINE = 100


class GeneticAlgorithm(GeneticMapping, BasicSolver):
    """GA: Genetic Algorithm"""

    def __init__(
        self,
        data_file_path: str,
        num_organisms: int = 50,
        coding_method: str = "binary",  # "binary" or "order"
        crossover_rate: float = 1,
        mutation_rate: float | None = 0.01,
        replace_rate: float = None,
        replace_rule: str = "parents",
        max_iteration: int = 100,
        no_print: bool = False,
        debug: bool = False
    ):
        """initialize the GA

        Args:
            data_file_path (str): data file
            num_organisms (int, optional): num of organs in population. Defaults to 50.
            coding_method (str, optional): code method. Defaults to "binary".
            crossover_rate (float): crossover rate.
            mutation_rate (float, optional): mutation rate.
            replace_rate (float, optional): initial replace rate, can control by a functio.
            replace_rule (str, optional): replace rule, 'worst', 'random', 'parens'.
            max_iteration (int): max iteration.
            no_print (bool, optional): print some info or not. Defaults to False.
            debug (bool, optional): print some debug info or not. Defaults to False.
        """
        # initialize the parent class
        GeneticMapping.__init__(self, data_file_path)
        BasicSolver.__init__(self, max_iteration)
        self.data_file_path = data_file_path
        self.max_iteration = max_iteration

        # coding method, binary or order
        self.coding_method = coding_method

        # other param
        self.num_organisms = num_organisms
        self.no_print = no_print
        self.debug = debug
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.replace_rate = replace_rate
        self.replace_rule = replace_rule

        # generation data
        self.population: List[np.ndarray] = []  # [theta, ...]
        self.f_vals: np.ndarray = np.array([], dtype=float)  # [func]
        self.fitnesses: np.ndarray = np.array([], dtype=float)  # [fitness]

        if not self.no_print:
            print("="*LOG_N_LINE)
            print("Genetic Algorithm Method")
            print(f"Sample data from : {self.data_file_path}")
            print(f"Max Iteration: {self.max_iteration}")

    def optim(self, replace_rate_func: callable, thetas: Optional[List[np.ndarray]] = None, elite_rate: float = None, num_crossover: int = 3, swap: int = 1, k_tournament: int = 4):
        """optim process

        Args:
            thetas (Optional[List[np.ndarray]], optional): list of solution. Defaults to None.
            elite_rate (float, optional): probability of if save the best one back to population. Defaults to not.
            replace_rate_func (callable): function control the replace rate
            num_crossover (int, optional): num of crossover points. Defaults to 3.
            swap (int, optional): swapping times. Defaults to 1.
            k_tournament (int, optional): divide population into k group and select parents.
        """
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print("Optim Process ...\n")

        # initialize
        self._initial_generation(thetas=thetas)

        # first fitness
        _, theta_opt, f_opt = self._fitness()
        self.record(0, theta=theta_opt, f_value=f_opt)  # record

        # replace rate
        replace_rate = self.replace_rate

        # optim process iter
        for iter in range(self.max_iteration):
            # evolution for this iter/population
            self._evolution(
                iter=iter,
                replace_rate=replace_rate,
                num_crossover=num_crossover,
                swap=swap,
                k_tournament=k_tournament
            )  # in self._evolution(): function self.record() has been applied

            # update replace rate
            replace_rate = replace_rate_func(replace_rate, iter)

            # print or not
            if self.debug:
                print(
                    f"Iter {iter}\tCurrent Func = {max(self.f_vals):.4f}\tOpt Func = {self.f_opt:.4f}"
                )
            else:
                if (not self.no_print) and (iter % max(1, self.max_iteration // 10) == 0):
                    print(
                        f"Iter {iter}\tCurrent Func = {max(self.f_vals):.4f}\tOpt Func = {self.f_opt:.4f}"
                    )
        # print final result
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print(f"Opt Solution: {list(self.theta_opt)}")
            print(f"Opt Objective Func: {self.f_opt}")
            print("="*LOG_N_LINE)

    def _evolution(self, iter: int = 0, replace_rate: float = None, elite_rate: float = None, num_crossover: int = 3, swap: int = 1, k_tournament: int = 4):
        """evolution: one iter, population changed

        Args:
            iter (int, optional): iter. Defaults to 0.
            replace_rate (float, optional): replace rate. Defaults to None.
            elite_rate (float, optional): probability of if save the best one back to population. Defaults to not.
            num_crossover (int, optional): num of crossover in one theta. Defaults to 3.
            swap (int, optional): num of swap, only for order coding. Defaults to 1.
            k_tournament (int, optional): divide population into k group and select parents.

        Raises:
            ValueError: initialize population first
            ValueError: replace method
        """
        if replace_rate is None:
            replace_rate = 1 / max(1, self.p)

        # random select new_num_pairs of parents to crossover and mutation, based on fitness
        # 0. defualt: we have _initial_generation
        if len(self.population) == 0:
            raise ValueError("Initialize the population first!")

        # 1. fitness -> selection prob
        fitnesses, _, _ = self._fitness()

        # 2. generate offspring
        # we need to update replace_rate * num_organisms by offspring
        new_num = int(replace_rate * self.num_organisms)
        new_num_pairs = new_num // 2
        num_parents = 2 * new_num_pairs

        # select parents by Tournament Selection
        parent_indices = self._tournament_selection(
            k=k_tournament, num_parents=num_parents
        )

        # record the offspring, for replacement
        offspring = []
        for i in range(0, len(parent_indices), 2):
            # simple use i, i+1
            idx1, idx2 = parent_indices[i], parent_indices[i + 1]
            parent1, parent2 = self.population[idx1], self.population[idx2]
            # crossover
            child1, child2 = self._crossover(
                parent1, parent2, num_crossover=num_crossover
            )
            # mutation
            child1 = self._mutation(child1, swap=swap)
            child2 = self._mutation(child2, swap=swap)
            # add into offspring list
            offspring.extend([child1, child2])

        # 3. replace
        n_offspring = len(offspring)
        rule = self.replace_rule.lower()  # 'worst', 'random', 'parents' method
        if rule == "worst":
            # replce min fitness
            replace_indices = np.argsort(fitnesses)[:n_offspring]
        elif rule == "random":
            # random replace
            replace_indices = np.random.choice(
                len(self.population), n_offspring, replace=False)
        elif rule == "parents":
            # replace their parents (make sure length equal)
            replace_indices = parent_indices[:n_offspring]
        else:
            raise ValueError(
                f"Unknown replace rule: {self.replace_rule}, please choose 'worst', 'random' or 'parents'")

        # apply replacement
        for i, idx in enumerate(replace_indices):
            self.population[idx] = offspring[i]

        # 4. do not delete the best one or not
        if elite_rate is not None and np.random.rand() < elite_rate:
            elite_idx = np.argmax(self.f_vals)
            elite = self.population[elite_idx].copy()
            # add the best one back, replace the worst one
            self.population[np.argmin(self.f_vals)] = elite

        # 5. calculate updated fitness, func value
        self._fitness()
        idx_opt = int(np.argmax(self.f_vals))

        # record
        self.record(
            iteration=iter,
            theta=self.population[idx_opt].copy(),
            f_value=self.f_vals[idx_opt]
        )

    def _tournament_selection(self, k, num_parents) -> List[int]:
        """Tournament Selection method for parents selection (with replacement)

        Args:
            k (_type_): k group
            num_parents (_type_): wanted number of parents

        Returns:
            List[int]: list of index of selected solution as parent
        """
        parent_indices = []
        # select until enough
        while len(parent_indices) < num_parents:
            # random shuffled
            random_idx = np.random.permutation(len(self.population))

            # split into k group
            subsets = np.array_split(random_idx, k)
            for subset in subsets:
                # find the best in this group
                idx_opt = subset[np.argmax(self.fitnesses[subset])]
                parent_indices.append(idx_opt)

                # enough number
                if len(parent_indices) >= num_parents:
                    break

        return parent_indices

    def _initial_generation(self, thetas: Optional[List[np.ndarray]] = None):
        """generate the first generation

        Args:
            thetas (Optional[List[np.ndarray]], optional): list of solution. Defaults to None.
        """
        if not thetas:
            # random initial generation
            if self.coding_method == "binary":  # 1. binary code
                self.population = [
                    np.random.randint(
                        low=0, high=2, size=self.p, dtype=int
                    ) for _ in range(self.num_organisms)
                ]  # (num_organisms, p), 0 or 1
            elif self.coding_method == "order":  # 2. order coding
                self.population = [
                    np.random.permutation(self.p).astype(int)
                    for _ in range(self.num_organisms)
                ]  # (num_organisms, p), 0 ~ p-1
            else:
                raise ValueError("Coding Method must be 'binary' or 'order'!")
        else:
            # generate the first from thetas
            for theta in thetas:
                self.population.append(self._valid_solution(theta=theta))

    def _fitness(self, population: Optional[List[Tuple[np.ndarray, float]]] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """calculate the fitnesses

        Args:
            population (Optional[List[Tuple[np.ndarray, float]]], optional): current population. Defaults to None.

        Returns:
            np.ndarray: [fitnesses, theta_opt, f_opt]
        """
        # use default
        if population is None:
            population = self.population

        # calculate the objective function
        self.f_vals = np.array([self.log_likelihood(theta)
                               for theta in population], dtype=float)

        # sort index: from large to small
        rank_idx = np.argsort(-self.f_vals)  # [4, 2, 3, ...]

        # opt solution
        theta_opt = self.population[rank_idx[0]]
        f_opt = self.f_vals[rank_idx[0]]

        # calculate the fitnesses
        self.fitnesses = np.zeros(self.num_organisms)
        for rank, idx in enumerate(rank_idx):
            real_rank = self.num_organisms - rank
            division = self.num_organisms * (self.num_organisms + 1)
            self.fitnesses[idx] = 2 * real_rank / division

        return self.fitnesses, theta_opt, f_opt

    def _crossover(self, theta1: Union[list, np.ndarray], theta2: Union[list, np.ndarray], num_crossover: int = 1) -> Tuple[np.ndarray]:
        """Crossover action

        Args:
            theta1 (Union[list, np.ndarray]): solution1
            theta2 (Union[list, np.ndarray]): solution1
            num_crossover (int, optional): crossover happened at `num_crossover` positions. Defaults to 1.

        Returns:
            Tuple[np.ndarray]: (new1, new2)
        """
        # no crossover
        if np.random.rand() >= self.crossover_rate:
            return theta1, theta2

        # valid theta
        theta1 = self._valid_solution(theta=theta1)
        theta2 = self._valid_solution(theta=theta2)

        child1, child2 = None, None

        # 1. crosover for binary coding
        if self.coding_method == "binary":
            for _ in range(num_crossover):
                pos = np.random.randint(1, self.p)  # 1, 2, ..., p-1
                child1 = np.concatenate((theta1[:pos], theta2[pos:]))
                child2 = np.concatenate((theta2[:pos], theta1[pos:]))

        # 2. crosover for order coding
        if self.coding_method == "order":
            # not larger than p, if = p => child = parent
            num_crossover = min(num_crossover, self.p - 1)

            # select the pos for exchange
            def _pos(num_crossover):
                # (3, 5, 7) -> the order of theta1 in position (3, 5, 7) copy to theta2
                return np.sort(
                    np.random.choice(
                        np.arange(self.p), size=num_crossover, replace=False
                    )
                )

            # exchange function
            def _crossover_with_pos_order(theta1, theta2, pos):
                """theta2 copy from theta1"""
                # 1. value of theta1 in position `pos`
                value_at_pos = theta1[pos]

                # 2. index of `value_at_pos` in theta2
                idx_of_value = np.where(np.isin(theta2, value_at_pos))[0]

                # 3. index of theta2 enter the value of theta1 in theta1's order
                child = theta2.copy()
                child[idx_of_value] = value_at_pos
                return child

            # theta2 copy order from theta1
            pos = _pos(num_crossover)
            child1 = _crossover_with_pos_order(theta1, theta2, pos)
            # theta1 copy order from theta2
            pos = _pos(num_crossover)
            child2 = _crossover_with_pos_order(theta2, theta1, pos)

        return child1, child2

    def _mutation(self, theta: Union[list, np.ndarray], swap: int = 1) -> np.ndarray:
        """mutation of theta

        Args:
            theta (Union[list, np.ndarray]): one solution
            swap (int, optional): as for order coding, swap times. Defaults to 1.

        Returns:
            np.ndarray: new theta
        """
        # valid theta
        theta = self._valid_solution(theta=theta)
        mutation_rate = self.mutation_rate

        if mutation_rate is None:
            return theta

        # mutate at probability `mutation_rate`
        # 1. binary coding, 0 <-> 1
        if self.coding_method == "binary":
            # (p,) from U(0, 1)
            rand_array = np.random.rand(self.p)
            mask = rand_array < mutation_rate  # whether mutation array
            new_theta = theta.copy()
            if np.any(mask):
                new_theta[mask] = 1 - new_theta[mask]  # 1 <-> 0
            return new_theta

        # 2. order coding
        if self.coding_method == "order":
            # random: select one pair (i, j) and swap it to (j, i)
            new_theta = theta.copy()
            for _ in range(swap):
                # random under p = mutation rate
                if np.random.rand() < mutation_rate:
                    i, j = np.random.choice(self.p, 2, replace=False)
                    new_theta[i], new_theta[j] = new_theta[j], new_theta[i]

            return new_theta

    def neighborhood(self):
        """As for Genetic Algorithm, we do not construct neighborhood"""
        pass
