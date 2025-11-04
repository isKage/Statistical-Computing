import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Union, Optional, List, Tuple


class BasicSolver(ABC):
    """Virtual class

    Args:
        ABC (_type_): Virtual class
    """

    def __init__(self, max_iteration: int = 100):
        """Virtual class initialize

        Args:
            max_iteration (int, optional): max iteration rounds. Defaults to 100.
        """
        # Opt among all historical solution
        self.theta_opt = None   # optimal x
        self.f_opt = -np.inf  # max f

        # history: [{"iter": int, "x": np.ndarray(), "f": objective function}]
        self.history = []
        self.max_iteration = max_iteration

    @abstractmethod
    def neighborhood(self, *args, **kwargs):
        """generate neighborhood"""
        pass

    @abstractmethod
    def optim(self, *args, **kwargs):
        """update method"""
        pass

    def record(self, iteration: int, theta: list | np.ndarray, f_value: float):
        """record function for other class to update history

        Args:
            iteration (int): current iter
            theta (list | np.ndarray): current solution
            f_value (float): current objective function
        """
        # valid theta
        if isinstance(theta, list):
            theta = np.ndarray(theta, dtype=int)

        # record history
        self.history.append(
            {"iter": iteration, "x": theta.copy(), "f": f_value}
        )

        # update f, max or min
        if f_value > self.f_opt:
            self.f_opt = f_value
            self.theta_opt = theta.copy()

    def plot(self, figure_file_path: Optional[str] = None, plot_type: str = "line"):
        """Plot convergence process

        Args:
            figure_file_path (Optional[str]): path for saving. Default to None.
            plot_type (str): plot type, 'line' or 'scatter'
        """
        # raise error if no history
        if not self.history:
            raise Exception("Error: No iteration history.")

        # get x and y for ploting
        iterations = [h["iter"] for h in self.history]  # x value
        f_values = [h["f"] for h in self.history]  # y value

        # opt util to now
        opt_util_to_now = np.maximum.accumulate(f_values)

        # plot: x = iter, y = [red: opt(x<=iter), black: opt(x=iter), blue dotted: opt*]
        plt.figure(figsize=(14, 10), dpi=200)

        if plot_type == "scatter":
            plt.scatter(iterations, f_values, color="black",
                        s=20, label="Current")
            plt.scatter(iterations, opt_util_to_now,
                        color="red", s=20, label="Opt so far")
        else:  # line plot (default)
            plt.plot(iterations, f_values, color="black", label="Current")
            plt.plot(iterations, opt_util_to_now,
                     color="red", label="Opt so far")

        # plt.plot(iterations, f_values, color="black", label="Current")
        # plt.plot(iterations, opt_util_to_now, color="red", label="Opt so far")

        plt.axhline(y=self.f_opt, color="blue",
                    linestyle="--", label="Final Opt")
        plt.xlabel("Iteration")
        plt.ylabel("Objective value")
        plt.title("Convergence Process")
        plt.legend()
        plt.grid(True)

        # save
        if figure_file_path:
            plt.savefig(figure_file_path, bbox_inches='tight')
            plt.close()
            # print info
            print(f"Figure saved in {figure_file_path}!")
        else:
            plt.show()
