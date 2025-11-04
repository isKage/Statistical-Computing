import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from .basic import BasicSolver
from .genetic_mapping import GeneticMapping

LOG_N_LINE = 100


class RandomCombinatorial(GeneticMapping, BasicSolver):
    def __init__(self, data_file_path, max_iteration=100, no_print: bool = True):
        GeneticMapping.__init__(self, data_file_path)
        BasicSolver.__init__(self, max_iteration)

        # do not print info
        self.no_print = no_print
        if not self.no_print:
            print("="*LOG_N_LINE)
            print("Random Combinatorial Method")
            print(f"Sample data from : {data_file_path}")
            print(f"Max Iteration: {max_iteration}")

    def neighborhood(self, theta: Union[list, np.ndarray], num: int = 5, swap: int = 1) -> np.ndarray:
        """Construct neighborhood

        Args:
            theta (Union[list, np.ndarray]): current solution, shape = (p,)
            num (int, optional): number of solution in neighborhood, including itself. Defaults to 5.
            swap (int, optional): swap times. Defaults to 1, means that every new solution, only swap 2 loc once.

        Returns:
            np.ndarray: neighborhood new solution, shape = (num, p)
        """
        # currnt solution: theta
        theta = self._valid_solution(theta=theta)

        # generate neighborhood
        neighborhood = []

        for _ in range(num):
            # generate num neighbors
            new_theta = theta.copy()
            # exchange swap times
            for _ in range(swap):
                i, j = np.random.choice(self.p, 2, replace=False)
                new_theta[i], new_theta[j] = new_theta[j], new_theta[i]
            neighborhood.append(new_theta)

        return np.array(neighborhood)

    def optim(self, theta_init: Union[list, np.ndarray], neighborhood_size=20):
        """optim

        Args:
            theta_init (Union[list, np.ndarray]): theta0, initial solution
            neighborhood_size (int, optional): num of neighbors. Defaults to 20.
        """
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print(f"Initial Solution: {theta_init}\nUpdating Process:\n")
        # initial
        theta = self._valid_solution(theta=theta_init)
        f_val = self.log_likelihood(theta)

        # record first solution
        self.record(0, theta, f_val)

        # begin iter
        for iter in range(1, self.max_iteration + 1):
            # 1. generate neighborhood
            neighbors = self.neighborhood(theta, num=neighborhood_size)

            # 2. calculate f in neighborhood
            fs = np.array([self.log_likelihood(th) for th in neighbors])

            # 3. select the best one
            idx_opt = np.argmax(fs)
            theta_opt, f_opt = neighbors[idx_opt], fs[idx_opt]

            # 4. update or not
            if f_opt > f_val:
                theta, f_val = theta_opt.copy(), f_opt

            # record in history
            self.record(iter, theta, f_val)

            # print info
            epoch = int(self.max_iteration / 10)
            if not self.no_print:
                if iter % epoch == 0:
                    print(
                        f"Iter: {iter}\tSolution: {list(theta)}\tObjective Func: {f_val:.4f}"
                    )
        if not self.no_print:
            print("-"*LOG_N_LINE)
            print(f"Opt Solution: {list(self.theta_opt)}")
            print(f"Opt Objective Func: {self.f_opt}")
            print("="*LOG_N_LINE)


class ParallelRandomCombin:
    """Multi-Process: parallel running"""

    def __init__(self, data_file_path: str, max_iteration: int = 100, works: Union[int, None] = None, neighborhood_size: int = 20):
        """Initialize

        Args:
            data_file_path (str): data file path
            max_iteration (int, optional): iteration rounds. Defaults to 100.
            works (int, optional): number of works. Defaults to None.
            neighborhood_size (int, optional): num of neighborhood. Defaults to 20.
        """
        self.data_file_path = data_file_path
        self.max_iteration = max_iteration
        self.neighborhood_size = neighborhood_size

        # works, e.g. = number of cpu cores
        if works:
            self.works = works
        else:
            self.works = os.cpu_count()

        self.results = []

        # print some infos
        print("="*LOG_N_LINE)
        print("Random Combinatorial Method (Multi Init Points)")
        print(f"Sample data from : {data_file_path}")
        print(f"Max Iteration: {max_iteration}")
        print(f"Num of Works: {self.works}")

    def _run_single(self, theta_init: np.ndarray, idx: int) -> Tuple[int, RandomCombinatorial]:
        """run single initial

        Args:
            theta_init (np.ndarray): initial solution
            idx (int): id for this optim

        Returns:
            Tuple[int, RandomCombinatorial]: (id, final solver)
        """
        # initialize the optim
        solver = RandomCombinatorial(
            self.data_file_path, max_iteration=self.max_iteration
        )
        # optimizing
        solver.optim(theta_init, neighborhood_size=self.neighborhood_size)
        # final ans and id
        return (idx, solver.theta_opt, solver.f_opt, solver.history)

    def optim(self, theta_inits: List[np.ndarray]) -> List[Tuple[int, np.ndarray, float, List[dict]]]:
        """multi-initial points optimizing

        Args:
            theta_inits (List[np.ndarray]): multi-initial points

        Returns:
            List[Tuple[int, np.ndarray, float, List[dict]]]: results
        """
        results = []

        # print some info
        print("-"*LOG_N_LINE)
        print(f"There are {len(theta_inits)} init points!")
        print(f"Updating Process:\n")

        # Multi Process
        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {
                executor.submit(self._run_single, theta, i): i
                for i, theta in enumerate(theta_inits)
            }

            # get result
            for future in as_completed(futures):
                try:
                    res_i = future.result()
                    results.append(res_i)
                    print(
                        f"Work {res_i[0]} done\tSolution {res_i[1]}\tObjective Func {res_i[2]:.4f}"
                    )
                except Exception as e:
                    print(f"Work {res_i[0]} error: {e}")

        # collect all results
        results.sort(key=lambda x: x[0])
        self.results = results
        print("\nAll works have been done!")
        print("-"*LOG_N_LINE)

        idx_result_opt = np.argmax([r[2] for r in results])  # largest f
        result_opt = results[idx_result_opt]
        print(f"Final Optimal:")
        print(
            f"Work {result_opt[0]}\tSolution {result_opt[1]}\tObjective Func {result_opt[2]:.4f}"
        )

    def plot(self, figure_file_path: str = None):
        """plot the convergence plot

        Args:
            figure_file_path (str, optional): saved path. Defaults to None.

        Raises:
            Exception: No results
        """
        # use optim first
        if len(self.results) == 0:
            raise Exception("Error: Apply optim() first!")

        # plot
        plt.figure(figsize=(14, 10), dpi=200)
        for idx, _, _, history in self.results:
            fs = [h["f"] for h in history]
            plt.plot(fs, label=f"Start {idx}")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        plt.title("Convergence Curves of Multiple Initial Points")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        print("-"*LOG_N_LINE)
        if figure_file_path:
            plt.savefig(figure_file_path, dpi=200)
            print(f"Figure saved at: {figure_file_path}!")
        print("="*LOG_N_LINE)
