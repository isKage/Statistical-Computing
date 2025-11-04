import os
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Type

LOG_N_LINE = 110


class ParallelSolver:
    """ParallelSolver, for any solver or optim"""

    def __init__(self, works: int = None, check: str = None):
        """initialize

        Args:
            works (int, optional): parallel works. Defaults to CPU cores.
        """
        self.works = works or os.cpu_count()
        self.results = []  # store history
        self.check = check  # check a key of kwargs

        print("=" * LOG_N_LINE)
        print(f"Parallel Solver Initialized: using {self.works} workers!")

    def _run_single(self, idx: int, solver_class: Type, init_kwargs: Dict[str, Any], optim_kwargs: Dict[str, Any]) -> tuple:
        """run a single optim

        Args:
            idx (int): id for find the optim
            solver_class (Type): solver class, e.g. Tabu, Random combin
            init_kwargs (Dict[str, Any]): param for __init__
            optim_kwargs (Dict[str, Any]): param for optim()

        Returns:
            tuple: some info of this task
        """
        # run the optim
        solver = solver_class(**init_kwargs)
        solver.optim(**optim_kwargs)

        # check the "check value", only care about the param in kwargs
        check_value = None
        if self.check:
            if self.check in init_kwargs:
                check_value = init_kwargs[self.check]
            elif self.check in optim_kwargs:
                check_value = optim_kwargs[self.check]

        return (idx, solver_class.__name__, solver.theta_opt, solver.f_opt, solver.history, check_value)

    def optim(self, task_list: List[Dict[str, Any]]):
        print("-" * LOG_N_LINE)
        print(f"Launching {len(task_list)} tasks in parallel...\n")

        with ProcessPoolExecutor(max_workers=self.works) as executor:
            futures = {
                executor.submit(
                    self._run_single,
                    i,  # idx
                    t["solver"],  # solver class
                    t["init_kwargs"],  # for __init__
                    t["optim_kwargs"],  # for optim()
                ): i for i, t in enumerate(task_list)
            }

            # get results
            for future in as_completed(futures):
                try:
                    res_i = future.result()
                    self.results.append(res_i)
                    # print info
                    print(
                        f"[Task {res_i[0]} | {res_i[1]}] Done\t{self.check}: {res_i[-1]}\tFunc = {res_i[3]:.4f}"
                    )
                except Exception as e:
                    print(f"[Task {res_i[0]} | {res_i[1]}] Error: {e}")

        print("\nAll tasks completed!")
        print("-" * LOG_N_LINE)
        # Now, check the final best
        final_opt = self.results[0]
        for res in self.results:
            if res[3] > final_opt[3]:
                final_opt = res
        print(f"Opt Solution: {final_opt[2]}")
        print(f"Opt Func: {final_opt[3]}")
        if self.check:
            print(f"{self.check}: {final_opt[-1]}")

    def plot(self, figure_file_path: str = None):
        # check history
        if not self.results:
            raise Exception("Error: No results yet, run optim() first.")

        # plot
        plt.figure(figsize=(14, 10), dpi=200)
        for idx, solver_name, _, _, history, check_value in self.results:
            fs = [h["f"] for h in history]
            fs_so_far = np.maximum.accumulate(fs)
            label_ = f"{solver_name} #{idx}"
            # put the check key and value on label
            if self.check:
                label_ += f" # {self.check}: {check_value}"
            plt.plot(fs_so_far, label=label_)
        plt.xlabel("Iteration")
        plt.ylabel("Objective Function")
        plt.title("Convergence Curves of All Parallel Tasks")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if figure_file_path:
            plt.savefig(figure_file_path, dpi=200)
            print(f"Figure saved: {figure_file_path}!")
        else:
            plt.show()
        print("="*LOG_N_LINE)
