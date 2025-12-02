import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_NUM = 70


class SIS:
    """SIS: Sequential Importance Sampling"""

    def __init__(self, N: int, T: int, Tmat: np.ndarray):
        """Initialize the SIS class

        Args:
            N (int): num of samples
            T (int): length of sequence
            Tmat (np.ndarray): transition matrix
        """
        # sample num
        self.N = N
        # length of seq
        self.T = T
        # Transition matrix
        self.Tmat = Tmat

        # result
        self.z_samples = None
        self.w = None

    def SIS(self, y: np.ndarray):
        """SIS process

        Args:
            y (np.ndarray): observed data
        """
        # store the result: all samples (N, T)
        self.z_samples = np.zeros((self.N, self.T), dtype=int)
        # store the result: log-weights (N, )
        logw = np.zeros(self.N)

        # 1. initialize z0 = 0
        z_prev = np.zeros(self.N, dtype=int)  # z0

        # start SIS
        for t in range(self.T):
            # 2. sample z_t ~ P(z_t | z_{t-1})
            p_zt = self.Tmat[z_prev, 1]
            z_t = (np.random.rand(self.N) < p_zt).astype(
                int)  # (T, F) -> (1, 0)

            self.z_samples[:, t] = z_t  # add to samples

            # 3. p(y_t | z_t)
            y_t = y[t]  # observed
            p_y = np.zeros(self.N)

            # fair dice
            idx_fair = (z_t == 1)
            p_y[idx_fair] = 1/6

            # loaded dice
            idx_loaded = (z_t == 0)
            p_y[idx_loaded & (y_t == 6)] = 1/2
            p_y[idx_loaded & (y_t != 6)] = 1/10

            # 4. update log-weights
            logw += np.log(p_y)

            # update previous state
            z_prev = z_t

        # 5. normalize weights
        _w = np.exp(logw - np.max(logw))
        self.w = _w / _w.sum()

    def summary(self, z_star: np.ndarray):
        """Summary

        Args:
            z_star (np.ndarray): actual hidden data
        """
        # ====================================
        # q1: estimate E(z|y)
        # ====================================
        z_hat = np.sum(self.w[:, None] * self.z_samples, axis=0)
        print("=" * LOG_NUM)
        print(f"E(z|y) estimation:\n{z_hat}\n")
        print(f"Efective Sample Size (ESS): {self.ESS():.4f}")
        print("-" * LOG_NUM)

        # ====================================
        # q2: arg max P(z|y)
        # ====================================
        idx_max = np.argmax(self.w)
        z_max = self.z_samples[idx_max]
        print(f"\narg max P(z|y):\n{z_max}")

        # compare z_max and z_star
        num_error = np.sum(z_max != z_star)
        print(f"\nError num: {num_error}")
        print(f"Accuracy:  {(1 - num_error/self.T) * 100:.1f}%")
        print("=" * LOG_NUM)

    def ESS(self):
        """calculate the Efective Sample Size (ESS)"""
        if abs(np.sum(self.w) - 1) < 1e-8:
            return 1 / np.sum(self.w**2)
        else:
            w = self.w / np.sum(self.w)
            return 1 / np.sum(w**2)


class SISParticleFilter(SIS):
    """SIS: Sequential Importance Sampling, with Particle Filter"""

    def __init__(self, N, T, Tmat, ess_threshold: float | int):
        """Initialize the SIS class

        Args:
            N (int): num of samples
            T (int): length of sequence
            Tmat (np.ndarray): transition matrix
            ess_threshold (float | int): ESS threshold, like N / 2
        """
        super().__init__(N, T, Tmat)
        self.ess_threshold = ess_threshold
        self.ess_list = []  # record the history of ess

    def SIS(self, y: np.ndarray):
        """rewrite SIS process, with Particle Filter

        Args:
            y (np.ndarray): observed data
        """
        # store the result: all samples (N, T)
        self.z_samples = np.zeros((self.N, self.T), dtype=int)
        # store the result: log-weights (N, )
        logw = np.zeros(self.N)

        # 1. initialize z0 = 0
        z_prev = np.zeros(self.N, dtype=int)  # z0

        # start SIS
        for t in range(self.T):
            # 2. sample z_t ~ P(z_t | z_{t-1})
            p_zt = self.Tmat[z_prev, 1]
            z_t = (np.random.rand(self.N) < p_zt).astype(
                int)  # (T, F) -> (1, 0)

            self.z_samples[:, t] = z_t  # add to samples

            # 3. p(y_t | z_t)
            y_t = y[t]  # observed
            p_y = np.zeros(self.N)

            # fair dice
            idx_fair = (z_t == 1)
            p_y[idx_fair] = 1/6

            # loaded dice
            idx_loaded = (z_t == 0)
            p_y[idx_loaded & (y_t == 6)] = 1/2
            p_y[idx_loaded & (y_t != 6)] = 1/10

            # 4. update weights
            logw += np.log(p_y)
            # normalize weights for ESS
            w = np.exp(logw - np.max(logw))
            w /= w.sum()
            self.w = w  # update weights

            # 5. check ESS
            ess = 1.0 / np.sum(w ** 2)
            self.ess_list.append(ess)
            if ess < self.ess_threshold:
                # 6. resample
                idx = SISParticleFilter._resample(w)
                self.z_samples = self.z_samples[idx]
                z_t = z_t[idx]

                # 7. update previous state
                z_prev = z_t  # next iteration will use

                # 8. reset weight
                w = np.ones(self.N) / self.N
                self.w = w
                logw = np.zeros(self.N)   # reset logw
            else:
                # 6. normal update previous state
                z_prev = z_t

    def _resample(w):
        """resampling"""
        N = len(w)
        positions = (np.arange(N) + np.random.rand()) / N

        indexes = np.zeros(N, dtype=int)
        cumsum = np.cumsum(w)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumsum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes

    def plot_ess(self, figure_file_path: str):
        plt.figure(figsize=(9, 5), dpi=200)
        plt.plot(self.ess_list)

        plt.title("ESS plot at sequence length = t")
        plt.xlabel("Sequence length: t")
        plt.ylabel("ESS")
        plt.grid(True)
        plt.savefig(figure_file_path)
