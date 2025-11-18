import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

from numpy.linalg import inv, slogdet

LOG_NUM = 50


class MultiNormal:
    """ECM Algorithm for Multiple-Multiple Normal Distribution Estimation"""

    def __init__(self, data: pd.DataFrame | np.ndarray, max_iter=10, eps=1e-8, no_print=False):
        """Initialize some hypeparameters

        Args:
            data (pd.DataFrame | np.ndarray): observered data
            max_iter (int, optional): max iteration rounds. Defaults to 10.
            eps (_type_, optional): precision. Defaults to 1e-8.
            no_print (bool, optional): print or not. Defaults to False.
        """
        # data
        if isinstance(data, pd.DataFrame):
            self.data = data.to_numpy()
        else:
            self.data = data
        self.n, self.d = data.shape

        self.max_iter = max_iter  # max iteration rounds
        self.eps = eps  # precision
        self.no_print = no_print  # print or not

        # record
        self.mu = None
        self.Sigma = None
        self.Q_history = []

    def _assistant_valiables(self, mu: np.ndarray, Sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assistant Valiables for update mu and Sigma

        Args:
            data (pd.DataFrame | np.ndarray): observered data
            mu (np.ndarray): mu^{(t)}, mu estimate
            Sigma (np.ndarray): Sigma^{(t)}, Sigma estimate

        Returns:
            Tuple[np.ndarray, np.ndarray]: S^{(1)}, S^{(2)}
        """
        n, d = self.data.shape
        S1 = np.zeros(d)
        S2 = np.zeros((d, d))

        for i in range(n):
            x = self.data[i]
            obs = ~np.isnan(x)
            mis = np.isnan(x)  # missing data

            if mis.sum() == 0:
                # if no missing, trivial
                u_hat = x
                r_hat = np.outer(x, x)
            else:
                # mis data | obs data
                x_o = x[obs]
                # mu, sigma -> mu_o, sigma_o, mu_m, sigma_m
                mu_o, mu_m = mu[obs], mu[mis]
                Sigma_oo = Sigma[np.ix_(obs, obs)]
                Sigma_om = Sigma[np.ix_(obs, mis)]
                Sigma_mo = Sigma[np.ix_(mis, obs)]
                Sigma_mm = Sigma[np.ix_(mis, mis)]

                # conditional mean & covariance
                inv_Soo = inv(Sigma_oo)
                mu_mo_cond = mu_m + Sigma_mo @ inv_Soo @ (x_o - mu_o)
                Sigma_mo_cond = Sigma_mm - Sigma_mo @ inv_Soo @ Sigma_om

                # E[U_i]
                u_hat = np.zeros(d)
                u_hat[obs] = x_o
                u_hat[mis] = mu_mo_cond

                # E[U_i U_i^T]
                r_hat = np.zeros((d, d))
                r_hat[np.ix_(obs, obs)] = np.outer(x_o, x_o)
                r_hat[np.ix_(obs, mis)] = np.outer(x_o, mu_mo_cond)
                r_hat[np.ix_(mis, obs)] = np.outer(mu_mo_cond, x_o)
                r_hat[np.ix_(mis, mis)] = Sigma_mo_cond + \
                    np.outer(mu_mo_cond, mu_mo_cond)

            S1 += u_hat
            S2 += r_hat

        return S1, S2

    def _update(self, S1: np.ndarray, S2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """CM Step, update the mu and Sigma

        Args:
            S1 (np.ndarray): sum of E(Ui | Uio)
            S2 (np.ndarray): sum of E(Ui Ui^T | Uio)
            n (int): _description_

        Returns:
            Tuple[np.ndarray, np.ndarray]: mu^{(t+1)}, Sigma^{(t+1)}
        """
        mu_new = S1 / self.n
        Sigma_new = (S2 - np.outer(S1, S1) / self.n) / self.n
        return mu_new, Sigma_new

    def Q_function(self, mu, Sigma):
        """Q function

        Args:
            mu (_type_): hat{mu}
            Sigma (_type_): hat{Sigma}

        Returns:
            float: Q function value
        """
        S1, S2 = self._assistant_valiables(mu=mu, Sigma=Sigma)
        mu = mu.copy().reshape(-1)  # (d,)
        S1 = S1.copy().reshape(-1)  # (d,)

        invS = np.linalg.inv(Sigma)  # Sigma^{-1}
        sign, logdet = np.linalg.slogdet(Sigma)  # log |Sigma|

        term1 = np.trace(invS @ S2)
        term2 = np.trace(invS @ np.outer(S1, mu))
        term4 = self.n * np.trace(invS @ np.outer(mu, mu))

        Q = -0.5 * self.n * logdet - 0.5 * (term1 - 2 * term2 + term4)
        return Q

    def optim(self, init_mu=None, init_Sigma=None):
        """optimization

        Args:
            init_mu (_type_, optional): Initial mu. Defaults to None.
            init_Sigma (_type_, optional): Initial Sigma. Defaults to None.
        """
        # initialize parameters
        if init_mu is None:
            # use mean of observered data
            init_mu = np.nanmean(self.data, axis=0)
        if init_Sigma is None:
            # simple fill the NA with mean
            df = pd.DataFrame(self.data)
            fill_values = {col: init_mu[i] for i, col in enumerate(df.columns)}
            data_filled = df.fillna(fill_values).to_numpy()
            # use sample covariance
            init_Sigma = np.cov(data_filled.T)

        mu = init_mu.copy()
        Sigma = init_Sigma.copy()

        print("=" * LOG_NUM)
        print("Iter\tQ Func\t\t|Curr Q - Prev Q|")
        for iter in range(self.max_iter):
            # E step: calculate the S1 and S2
            S1, S2 = self._assistant_valiables(mu, Sigma)
            # CM step: update the mu and Sigma
            mu_new, Sigma_new = self._update(S1, S2)

            # Q function
            Q_value = self.Q_function(mu_new, Sigma_new)

            # check convergence
            diff = None
            if len(self.Q_history) > 0:
                diff = abs(self.Q_history[-1] - Q_value)
                if diff < self.eps:
                    break

            # record
            self.Q_history.append(Q_value)

            # update
            mu, Sigma = mu_new, Sigma_new

            if not self.no_print and iter % (int(self.max_iter / 10)) == 0:
                if diff is None:
                    print(f"{iter}\t{Q_value:.4f}\tNone")
                else:
                    print(f"{iter}\t{Q_value:.4f}\t\t{diff:.5f}")

        # record the result
        self.mu = mu
        self.Sigma = Sigma
        print("-" * LOG_NUM)
        print(f"# Final Q = {Q_value}")
        mu_str = ", ".join([f"{v:.4f}" for v in mu])
        print(f"# Final mu = \n[{mu_str}]")
        Sigma_str = np.array2string(Sigma, precision=4, suppress_small=False)
        print(f"# FIinal Sigma = \n{Sigma_str}")
        print("=" * LOG_NUM)

    def plot(self, figure_file_path: str):
        """Plot Q function convergence plot"""
        Q_history = self.Q_history
        if Q_history is None:
            raise ValueError("Apply optim() first!")

        hist = np.array(Q_history)

        plt.figure(figsize=(10, 6), dpi=150)
        plt.plot(hist, label="Q Func")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Q Func value")
        plt.tight_layout()
        plt.savefig(figure_file_path)
        plt.close()
        
    ######################################################################
    #  新增方法：问题 (c) —— 均值填充后直接求 MLE
    ######################################################################

    def fit_mean_impute_mle(self, df, S1, S2):
        data = df.copy()
        col_means = data.mean(axis=0)
        filled = data.fillna(col_means).to_numpy()
        n, d = filled.shape

        # MLE
        mu_hat = filled.mean(axis=0)
        Xc = filled - mu_hat
        Sigma_hat = (Xc.T @ Xc) / n

        # Q value (COMPARE this with EM result)
        Q_mle = self.Q_function(mu_hat, Sigma_hat, S1, S2, n)

        return mu_hat, Sigma_hat, Q_mle
