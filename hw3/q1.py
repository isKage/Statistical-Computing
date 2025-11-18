import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import HIV, HIVCov

np.random.seed(42)

OUTPUT_DIR = os.path.join("figure")

data = [379, 299, 222, 145, 109, 95, 73, 59, 45, 30, 24, 12, 4, 2, 0, 1, 1]
LOG_NUM = 100

data_array = np.array(data)
prob = data_array / np.sum(data_array)


def main():
    # 1. initialize the class
    hiv = HIV(data=data)
    print("=" * LOG_NUM)
    print("##\tInit Params\t\t\t\tFinal Params\t\t\t\tFinal Q Func")

    # 2. initial points
    Rounds = 10
    results = []
    for r in range(Rounds):
        theta_init = np.random.uniform(size=4)
        theta_init_str = ", ".join([f"{v:.4f}" for v in theta_init])

        # 3. optim
        theta, Q_func = hiv.optim(
            theta_init=theta_init, epsilon=1e-6, max_iter=15)
        theta_str = ", ".join([f"{v:.4f}" for v in theta])
        print(
            f"{r+1}\t[{theta_init_str}]\t[{theta_str}]\t{Q_func:.4f}"
        )
        results.append((theta, Q_func, hiv.Q_history))

    # 3. compare and find the best
    best_theta, best_Q, _ = max(results, key=lambda x: x[1])
    print("-" * LOG_NUM)
    best_str = ", ".join([f"{v:.4f}" for v in best_theta])
    print(f"Best Params: [{best_str}]\nBest Q: {best_Q:.4f}")
    print("=" * LOG_NUM, "\n")

    # 4. plot
    plt.figure(figsize=(8, 6), dpi=200)
    for r, (_, _, hist_q) in enumerate(results):
        plt.plot(hist_q, label=f"Start {r+1}")

    plt.xlabel("Iteration")
    plt.ylabel("Q Function")
    plt.title("All Multi-Start EM Q Curves")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "Multi_HIV_EM_Plot.png"))
    plt.close()

    # figure_file_path = os.path.join(OUTPUT_DIR, "HIV_EM_Params_Plot.png")
    # hiv.plot_params(figure_file_path=figure_file_path)

    # figure_file_path = os.path.join(OUTPUT_DIR, "HIV_EM_Plot.png")
    # hiv.plot(figure_file_path=figure_file_path)

    hivcov = HIVCov(data=data, theta=best_theta)
    # ===== Cov =====
    COV_Params = hivcov.COV_Params(sample_size=30)
    COV_Params_str = np.array2string(
        COV_Params, precision=2, suppress_small=False)
    print("Cov = \n", COV_Params_str)

    # SE
    se = np.sqrt(np.diag(COV_Params))
    se_str = np.array2string(se, precision=2, suppress_small=False)
    print("SE = ")
    print(se_str)

    # ===== Corr =====
    Corr_Params = hivcov.Corr_Params(sample_size=30)
    COrr_Params_str = np.array2string(
        Corr_Params, precision=2, suppress_small=False)
    print("Corr = \n", COrr_Params_str)


def single():
    hiv = HIV(data=data)
    theta_init = [0.2, 0.6, 2, 3]
    theta, Q_func = hiv.optim(theta_init=theta_init, epsilon=1e-6, max_iter=15)
    print("# Final Solution:\n", theta)
    print("# Final Q Function: \n", Q_func)

    hivcov = HIVCov(data=data, theta=theta)
    # ===== Cov =====
    COV_Params = hivcov.COV_Params(sample_size=30)
    COV_Params_str = np.array2string(
        COV_Params, precision=4, suppress_small=False)
    # print("# Cov = \n", COV_Params_str)

    # SE
    se = np.sqrt(np.diag(COV_Params))
    se_str = np.array2string(se, precision=2, suppress_small=False)
    print("# SE = ")
    print(se_str)

    # ===== Corr =====
    Corr_Params = hivcov.Corr_Params(sample_size=30)
    COrr_Params_str = np.array2string(
        Corr_Params, precision=4, suppress_small=False)
    print("# Corr = \n", COrr_Params_str)


def bootstrap():
    Rounds = 20
    theta_list = []
    for _ in range(Rounds):

        # 1. bootstrap sampling
        sampled_idx = np.random.choice(
            len(data), size=np.sum(data_array), replace=True, p=prob)

        # 2. new data
        bootstrap_data = np.bincount(sampled_idx, minlength=len(data))

        # 3. optim
        hiv = HIV(data=bootstrap_data)
        theta_init = [0.2, 0.6, 2, 3]
        theta, _ = hiv.optim(theta_init=theta_init, epsilon=1e-6, max_iter=15)

        theta_list.append(theta)

    # -------- Cov --------
    theta_array = np.vstack(theta_list)  # shape (Rounds, 4)
    Cov_theta = np.cov(theta_array, rowvar=False)
    Cov_theta_str = np.array2string(
        Cov_theta, precision=4, suppress_small=False)
    print("# Cov = \n", Cov_theta_str)

    std = np.sqrt(np.diag(Cov_theta))
    se_str = np.array2string(std, precision=2, suppress_small=False)
    print("# SE = \n", se_str)

    Corr = Cov_theta / np.outer(std, std)
    Corr_str = np.array2string(
        Corr, precision=4, suppress_small=False)
    print("# Corr = \n", Corr_str)


if __name__ == "__main__":
    main()
    print("\n", "="*LOG_NUM, "\n")
    single()
    print("\n", "="*LOG_NUM, "\n")
    bootstrap()
