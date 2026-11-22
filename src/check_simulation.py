"""Simulation test for Project 3 (Iteration 3).

Runs the from-scratch SVD in methods.py on the Iteration 2 simulation
(120 x 80, rank 5 plus noise 0.01, seed 0) and checks it in three ways:

  Part 1. Against numpy.linalg.svd: sigma_i, U, V (signs aligned), tail sum.   (Q3, Q4)
  Part 2. Against the truth L (noise-free rank-5 matrix), for three sizes,
          with a log-log plot of the error.
  Part 3. Stress test: ask for k = 8.

numpy.linalg is used here only to check the answers, never inside methods.py.

Run:  python src/check_simulation.py
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from methods import my_svd_power, my_rank_k
from simulate import simulate_low_rank


def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e} rel = {rel_err:.2e}")


def part1_library():
    """Part 1: compare with numpy.linalg.svd (Q3, Q4) and check the tail-sum identity."""
    print("\n=== Part 1: from-scratch SVD vs numpy.linalg.svd ===")
    k = 5

    A = simulate_low_rank(m=120, n=80, r=5, noise=0.01, seed=0)

    sig, U, V = my_svd_power(A, k=k)
    U_lib, s_lib, Vt_lib = np.linalg.svd(A, full_matrices=False)
    
    report("sigma_1..5 vs library", sig, s_lib[:k])

    sgn = np.sign(np.sum(U * U_lib[:, :k], axis=0))   # +1 or -1 per column
    report("U (signs aligned)", U * sgn, U_lib[:, :k])
    report("V (sgn aligned)", V * sgn,  Vt_lib[:k].T)             # same sgn: u_i and v_i flip together

    A_k = my_rank_k(sig, U, V, k)
    print(np.sum((A - A_k)**2), np.sum(s_lib[k:]**2))
    report("tail sum vs sum sigma_i^2", np.sum((A - A_k)**2), np.sum(s_lib[k:]**2))

def part2_truth():
    """Part 2: compare with the truth L for three sizes; log-log plot of the error."""
    print("\n=== Part 2: from-scratch SVD vs the truth ===")
    sizes = [(60, 40), (120, 80), (480, 320), (1200, 800)]
    k = 5

    ms, errs, errs_raw = [], [], []
    for m, n in sizes:
        A, L = simulate_low_rank(m=m, n=n, r=5, noise=0.01, seed=0, return_truth=True)
        sig, U, V = my_svd_power(A, k=k)
        A_k = my_rank_k(sig, U, V, k)
        err = np.linalg.norm(A_k - L) / np.linalg.norm(L)
        err_raw = np.linalg.norm(A - L) / np.linalg.norm(L)
        ms.append(m); errs.append(err); errs_raw.append(err_raw)
        print(f"{m:5d} x {n:<4d}  ||A_k - L||/||L|| = {err:.2e}   ||A - L||/||L|| = {err_raw:.2e}")

    slope = np.polyfit(np.log(ms), np.log(errs), 1)[0]
    print(f"log-log slope = {slope:.2f}   (expected about -0.5)")

    plt.figure()
    plt.loglog(ms, errs, "o-", label="Rank-5 approximation")
    plt.loglog(ms, errs_raw, "s-", label="Noisy input")
    plt.xlabel("m")
    plt.ylabel("relative error")
    plt.legend()
    Path("figures").mkdir(exist_ok=True)
    plt.savefig("figures/sim_error_vs_size.png", dpi=300, bbox_inches="tight")
    plt.close()

def part3_stress():
    """Part 3: stress test, k = 8 (noise singular values sigma_6, sigma_7, ... are almost equal)."""
    print("\n=== Part 3: stress test, k = 8 ===")
    k = 8
    tol = 1e-12
    A = simulate_low_rank()
    _, s_lib, Vt_lib = np.linalg.svd(A, full_matrices=False)
    V_lib = Vt_lib[:k].T

    ratio = s_lib[1:k + 1] / s_lib[:k]
    iters = np.ceil(np.log(tol) / np.log(ratio**2)).astype(int)
    print("sigma_1..9 =", s_lib[:k + 1])
    print("sigma_(i+1) / sigma_i =", ratio)
    print("estimated iterations for v_1..8 =", iters)

    for mi in (10, 50, 200, 1000, 5000):
        _, _, V = my_svd_power(A, k, tol=tol, max_iter=mi)
        sgn = np.sign(np.sum(V * V_lib, axis=0))
        col_err = np.abs(V * sgn - V_lib).max(axis=0)
        noise_errors = ", ".join(f"{err:.2e}" for err in col_err[5:])
        print(
            f"max_iter = {mi:4d}  max error v_1..5 = {col_err[:5].max():.2e}"
            f"  errors v_6..8 = [{noise_errors}]"
        )


if __name__ == "__main__":
    part1_library()
    part2_truth()
    part3_stress()
