import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional, List, Tuple


class GeneticMapping:
    """Genetic Mapping: for storing the info of this mapping"""

    def __init__(self, data_file_path: str):
        """Initialize the GeneticMapping class

        Args:
            data_file_path (str): sample data file path

        Raises:
            FileNotFoundError: cannot find the data
        """
        # load data
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"File not found: {data_file_path}")
        self.data = pd.read_csv(data_file_path)

        # nrow, ncol and col's name
        self.n, self.p = self.data.shape
        self.loc_names = list(self.data.columns)

    def genetic_distance(self, theta: Union[list, np.ndarray, None] = None) -> np.ndarray:
        """estimate the d(theta_j, theta_j+1)

        Args:
            theta (Union[list, np.ndarray, None], optional): a solution (note the location of every gene loc), shape: (p,)

        Returns:
            np.ndarray: probabiltiy of every gene loc, shape: (p-1,)
        """
        # Validation solution
        theta = self._valid_solution(theta=theta)

        # replace
        data = self.data.iloc[:, theta]

        # hat{d} = mean(|loci+1 - loci|), shape = (p-1,)
        d_hat = data.diff(axis=1).iloc[:, 1:].abs().mean()

        return np.array(d_hat)

    def log_likelihood(self, theta: Union[list, np.ndarray, None] = None) -> float:
        """log likelihood, objective function (numpy for accelerate)

        Args:
            theta (Union[list, np.ndarray, None], optional): a solution. Defaults to None.

        Returns:
            float: log(L(theta;x))
        """
        # Validation solution
        theta = self._valid_solution(theta=theta)

        # d estimate
        d_hat = self.genetic_distance(theta=theta)

        # calculate the log likelihood
        return self.n * (
            d_hat.T @ np.log(d_hat) + (1-d_hat).T @ np.log(1-d_hat)
        )

    def _valid_solution(self, theta: Union[list, np.ndarray, None] = None) -> np.ndarray:
        # as for a solution theta
        if theta is None:
            theta = np.arange(self.p, dtype=int)
        else:
            theta = np.array(theta, dtype=int)
        # if loc range not from 0
        if max(theta) > self.p - 1:
            theta = theta - 1
        return theta
