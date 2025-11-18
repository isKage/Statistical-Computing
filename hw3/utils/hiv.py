import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union


class HIV:
    """EM Algorithm for HIV Case Estimation"""

    def __init__(self, data: Optional[List | np.ndarray], EPS=1e-8):
        """Initialize some hypeparameters

        Args:
            data (Optional[List  |  np.ndarray]): Observered Data
            EPS (_type_, optional): precision. Defaults to 1e-8.
        """
        self.data = np.array(data)
        self.C = len(data)
        self.EPS = EPS

        self.theta = None
        self.history = None
        self.Q_history = None

    def _pii(self, i: int, alpha, beta, mu, lam):
        """pi: probability"""
        pii = beta * mu**i * np.exp(-mu) + \
            (1 - alpha - beta) * lam**i * np.exp(-lam)
        if i == 0:
            pii = pii + alpha
        return pii

    def _z0(self, alpha, beta, mu, lam):
        """z0(theta)"""
        pii = self._pii(0, alpha, beta, mu, lam)
        return alpha / max(pii, self.EPS)

    def _ti(self, i: int, alpha, beta, mu, lam):
        """ti(theta)"""
        pii = self._pii(i, alpha, beta, mu, lam)
        return beta * mu**i * np.exp(-mu) / max(pii, self.EPS)

    def _pi(self, i: int, alpha, beta, mu, lam):
        """pi(theta)"""
        pii = self._pii(i, alpha, beta, mu, lam)
        return (1 - alpha - beta) * lam**i * np.exp(-lam) / max(pii, self.EPS)

    def _Q(self, alpha, beta, mu, lam):
        """Q function"""
        # not too small
        EPS = 1e-30
        alpha = max(alpha, EPS)
        beta = max(beta,  EPS)
        mu = max(mu,  EPS)
        lam = max(lam, EPS)

        T = P = iT = iP = 0.0
        for i in range(self.C):
            ti = self._ti(i, alpha, beta, mu, lam)
            pi = self._pi(i, alpha, beta, mu, lam)
            T += self.data[i] * ti
            P += self.data[i] * pi
            iT += i * self.data[i] * ti
            iP += i * self.data[i] * pi

        # ----- Q function -----
        Q = 0.0
        Q += self.data[0] * self._z0(alpha, beta, mu, lam) * np.log(alpha)
        Q += T * np.log(beta)
        Q += P * np.log(max(1 - alpha - beta, EPS))
        Q += iT * np.log(mu) - T * mu
        Q += iP * np.log(lam) - P * lam

        return Q

    def optim(self, theta_init: Optional[List | np.ndarray], epsilon=1e-6, max_iter=100):
        """Optimization

        Args:
            theta_init (Optional[List  |  np.ndarray]): Initial points of params
            epsilon (_type_, optional): precision. Defaults to 1e-6.
            max_iter (int, optional): max iteration rounds. Defaults to 100.

        Returns:
            _type_: best solution and best Q function
        """
        # intial point of parameters
        if theta_init is None:
            theta_init = np.random.uniform(size=4)
        else:
            theta_init = np.array(theta_init)

        theta = theta_init.copy()

        N = np.sum(self.data)
        self.history = []
        self.Q_history = []
        for _ in range(max_iter):
            alpha, beta, mu, lam = theta

            # ---------- E step ----------
            z0 = self._z0(alpha, beta, mu, lam)

            T = P = iT = iP = 0.0
            for i in range(self.C):
                ti = self._ti(i, alpha, beta, mu, lam)
                pi = self._pi(i, alpha, beta, mu, lam)
                T += self.data[i] * ti
                P += self.data[i] * pi
                iT += i * self.data[i] * ti
                iP += i * self.data[i] * pi

            # ---------- M step ----------
            alpha_new = self.data[0] * z0 / N
            beta_new = T / N
            # not divided by zero
            mu_new = iT / T if T > 1e-12 else mu
            lam_new = iP / P if P > 1e-12 else lam

            # not too small or big
            alpha_new = np.clip(alpha_new, self.EPS, 1 - self.EPS)
            beta_new = np.clip(beta_new, self.EPS, 1 -
                               alpha_new - self.EPS)  # alpha + beta < 1

            # new solution
            theta_new = np.array([alpha_new, beta_new, mu_new, lam_new])
            # record
            self.history.append(theta_new.copy())
            self.Q_history.append(
                self._Q(alpha_new, beta_new, mu_new, lam_new)
            )
            # print(f"{_}", self.Q_history[-1])

            # convergence
            if np.max(np.abs(theta_new - theta)) < epsilon:
                self.theta = theta_new
                return theta_new, self.Q_history[-1]

            # updated
            theta = theta_new

        self.theta = theta  # saved
        return self.theta, self.Q_history[-1]

    def plot_params(self, figure_file_path: str):
        """Plot params convergence plot"""
        # use self.history plot 4 line: alpha, beta, mu, lam
        history = self.history
        if history is None:
            raise ValueError("Apply optim() first!")

        hist = np.array(history)
        names = ["alpha", "beta", "mu", "lambda"]

        plt.figure(figsize=(10, 6), dpi=150)
        for k in range(4):
            plt.plot(hist[:, k], label=names[k])
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Parameter value")
        plt.tight_layout()
        plt.savefig(figure_file_path)
        plt.close()

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


class HIVCov(HIV):
    """Louis Method for Estimating the Cov(theta)"""

    def __init__(self, data: Optional[List | np.ndarray], theta: Optional[List | np.ndarray], EPS=1e-8):
        """Initialize some hypeparameters

        Args:
            data (Optional[List  |  np.ndarray]): observered data
            theta (Optional[List  |  np.ndarray]): best solution
            EPS (float, optional): precision. Defaults to 1e-8.
        """
        super().__init__(data=data)

        try:
            self.theta = np.array(theta)
        except:
            ValueError("Input final solution!")

    def COV_Params(self, sample_size=30):
        """Cov(theta)

        Args:
            sample_size (int, optional): Monte Canlo Method Estimation. Defaults to 30.

        Returns:
            np.ndarray: Cov(theta)
        """
        iY = self.iY()
        iZX = self.iZX(sample_size=sample_size)
        try:
            COV = np.linalg.inv(iY - iZX)
        except np.linalg.LinAlgError:
            COV = np.linalg.pinv(iY - iZX)
        return COV

    def Corr_Params(self, sample_size=30):
        COV = self.COV_Params(sample_size=sample_size)
        std = np.sqrt(np.diag(COV))
        Corr = COV / np.outer(std, std)
        return Corr

    def iY(self):
        """iY: fisher information estimation of complete data"""
        # initialize iY with 0
        dim_of_params = len(self.theta)
        iY = np.zeros(shape=(dim_of_params, dim_of_params))

        # final solution
        alpha, beta, mu, lam = self.theta

        # calculate some variables
        z0 = self._z0(alpha, beta, mu, lam)
        T = P = iT = iP = 0.0
        for i in range(self.C):
            ti = self._ti(i, alpha, beta, mu, lam)
            pi = self._pi(i, alpha, beta, mu, lam)
            T += self.data[i] * ti
            P += self.data[i] * pi
            iT += i * self.data[i] * ti
            iP += i * self.data[i] * pi

        # var(alpha | Y)
        iY[0][0] = self.data[0] * z0 / max(alpha**2, self.EPS) \
            + P / max((1-alpha-beta)**2, self.EPS)
        # var(beta | Y)
        iY[1][1] = T / max(beta**2, self.EPS) \
            + P / max((1-alpha-beta)**2, self.EPS)
        # var(mu | Y)
        iY[2][2] = iT / max(mu**2, self.EPS)
        # var(lambda | Y)
        iY[3][3] = iP / max(lam**2, self.EPS)

        # cov(alpha, beta | Y)
        iY[0][1] = iY[1][0] = P / max((1-alpha-beta)**2, self.EPS)

        return iY

    def iZX(self, sample_size=30):
        """iY: conditional fisher information estimation of missing data given observered data"""
        # final solution
        alpha, beta, mu, lam = self.theta

        # calculate some variables
        z0 = self._z0(alpha, beta, mu, lam)
        ti = np.array([self._ti(i, alpha, beta, mu, lam)
                      for i in range(self.C)])
        pi = np.array([self._pi(i, alpha, beta, mu, lam)
                      for i in range(self.C)])

        # estimate the var
        dlog = np.empty((sample_size, len(self.theta)))
        for m in range(sample_size):
            dlog[m, :] = self._d_log_f_ZX(z0=z0, ti=ti, pi=pi)

        return np.cov(dlog, rowvar=False, ddof=1)  # shape = (4, 4)

    def _d_log_f_ZX(self, z0, ti, pi):
        """d log f(Z|X) / d theta"""
        alpha, beta, mu, lam = self.theta
        dz0, dti, dpi, pi_vec, A, B = self._grad_z0_ti_pi(alpha, beta, mu, lam)

        # sample Z
        nz0 = np.random.binomial(self.data[0], z0)

        # sample T and P
        nt = np.random.binomial(self.data, ti)
        npv = np.random.binomial(self.data, pi)

        # z0
        z0_safe = np.clip(z0, self.EPS, 1 - self.EPS)
        ti_safe = np.clip(ti, self.EPS, 1 - self.EPS)
        pi_safe = np.clip(pi_vec, self.EPS, 1 - self.EPS)

        # A1
        A1 = nz0 / z0_safe - (self.data[0] - nz0) / (1 - z0_safe)

        # Bi and Ci vectors
        Bi = nt / ti_safe - (self.data - nt) / (1 - ti_safe)
        Ci = npv / pi_safe - (self.data - npv) / (1 - pi_safe)

        # accumulate score: start with z0 part
        res = A1 * dz0.copy()    # shape (4,)

        # add sum_i Bi * dti[i]  and Ci * dpi[i]
        # dti is (C,4), Bi is (C,)
        # do weighted sum over rows
        res += (Bi[:, None] * dti).sum(axis=0)
        res += (Ci[:, None] * dpi).sum(axis=0)

        return res  # shape (4,)

    def _grad_z0_ti_pi(self, alpha, beta, mu, lam):
        """Coding by AI: just for calculate the df / dx"""
        C = self.C
        i_idx = np.arange(C)

        # mu^i e^{-mu}, lambda^i e^{-lambda}
        mu_pow = np.power(mu, i_idx) * np.exp(-mu)
        lam_pow = np.power(lam, i_idx) * np.exp(-lam)

        # pi_i
        pi = (alpha * (i_idx == 0).astype(float)
              + beta * mu_pow
              + (1 - alpha - beta) * lam_pow)   # shape (C,)

        # --- dz0 ---
        pi0 = pi[0]
        dpi0_dalpha = 1.0 - np.exp(-lam)
        dpi0_dbeta = np.exp(-mu) - np.exp(-lam)
        dpi0_dmu = -beta * np.exp(-mu)
        dpi0_dlam = -(1 - alpha - beta) * np.exp(-lam)
        dpi0 = np.array([dpi0_dalpha, dpi0_dbeta, dpi0_dmu, dpi0_dlam])
        e_alpha = np.array([1.0, 0.0, 0.0, 0.0])
        dz0 = (e_alpha * pi0 - alpha * dpi0) / (pi0 * pi0)

        # --- dA_i (A=beta mu^i e^{-mu}), dB_i (B=(1-alpha-beta) lam^i e^{-lam}) ---
        dA_dalpha = np.zeros(C)
        dA_dbeta = mu_pow
        with np.errstate(divide='ignore', invalid='ignore'):
            mu_pow_minus1 = np.where(
                i_idx == 0, 0.0, np.power(mu, i_idx - 1) * np.exp(-mu))
        dA_dmu = beta * mu_pow_minus1 * (i_idx - mu)
        dA_dlam = np.zeros(C)

        # dB partials
        dB_dalpha = -lam_pow
        dB_dbeta = -lam_pow
        with np.errstate(divide='ignore', invalid='ignore'):
            lam_pow_minus1 = np.where(
                i_idx == 0, 0.0, np.power(lam, i_idx - 1) * np.exp(-lam))
        dB_dmu = np.zeros(C)
        dB_dlam = (1 - alpha - beta) * lam_pow_minus1 * (i_idx - lam)

        # dPi partials
        dpi_dalpha = (i_idx == 0).astype(float) + \
            dB_dalpha  # = 1_{i=0} - lam^i e^{-lam}
        dpi_dbeta = dA_dbeta + dB_dbeta  # = mu^i e^{-mu} - lam^i e^{-lam}
        dpi_dmu = dA_dmu
        dpi_dlam = dB_dlam
        dpi_all = np.stack([dpi_dalpha, dpi_dbeta, dpi_dmu, dpi_dlam], axis=1)

        # Now compute d(t_i) and d(p_i) using quotient rule:
        A = beta * mu_pow
        B = (1 - alpha - beta) * lam_pow

        # avoid zero pi
        pi_safe = np.clip(pi, self.EPS, None)
        pi_sq = pi_safe * pi_safe

        # derivatives of A and B stacked
        dA_all = np.stack(
            [dA_dalpha, dA_dbeta, dA_dmu, dA_dlam], axis=1)  # (C,4)
        dB_all = np.stack(
            [dB_dalpha, dB_dbeta, dB_dmu, dB_dlam], axis=1)  # (C,4)

        # t_i derivatives: (dA * pi - A * dpi) / pi^2
        dti = (dA_all * pi_safe[:, None] -
               (A[:, None] * dpi_all)) / (pi_sq[:, None])
        # p_i derivatives: (dB * pi - B * dpi) / pi^2
        dpi = (dB_all * pi_safe[:, None] -
               (B[:, None] * dpi_all)) / (pi_sq[:, None])

        return dz0, dti, dpi, pi, A, B  # return also pi, A, B for reuse
