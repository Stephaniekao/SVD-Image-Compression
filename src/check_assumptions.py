"""Check the four Iteration 2 assumptions, on the simulation first, then on the real image.

The simulation is low rank by construction, so its checks show what "passing" looks like.
  1. The image is close to low rank.
  2. Error is measured by the Frobenius norm.
  3. Singular values are sorted and u_i is computed from v_i.
  4. Computing through A^T A is accurate enough.

Run:  python src/check_assumptions.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis import load_image
from simulate import simulate_low_rank
from methods import my_svd_power, my_rank_k, my_energy

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "figures"


def check(name, A, k_max, ks):
    """Run the four checks on one data matrix.

    Input:  name (str), A (m x n), k_max (number of singular values to compute),
            ks (values of k to print, each < k_max).
    Output: sig (ours, length k_max), s_lib (library, all), rel (relative error of each sigma_i).
    """
    m, n = A.shape
    total = np.sum(A ** 2)
    sig, U, V = my_svd_power(A, k=k_max)
    s_lib = np.linalg.svd(A, compute_uv=False)          # check only

    print(f"\n=== {name}: {m} x {n}, k_max = {k_max} ===")

    # 1. Low rank: energy(k) and the gap sigma_k / sigma_{k+1}
    energy = my_energy(sig, A)
    print("1. Low rank")
    for k in ks:
        print(f"   k = {k:3d}: energy = {energy[k - 1]:.6f}, "
              f"sigma_k / sigma_(k+1) = {sig[k - 1] / sig[k]:.2f}")

    # 2. Frobenius error ||A - A_k||_F / ||A||_F vs the worst single entry
    print("2. Frobenius error vs worst entry")
    for k in ks:
        R = A - my_rank_k(sig, U, V, k)
        print(f"   k = {k:3d}: ||A - A_k||_F / ||A||_F = {np.sqrt(np.sum(R ** 2) / total):.4f}, "
              f"max |a_ij - (A_k)_ij| = {np.abs(R).max():.4f}")

    # 3. Sorted: sigma_1 >= sigma_2 >= ...;  paired: A v_i = sigma_i u_i;  orthonormal: U^T U = I
    #    (u_i = A v_i / sigma_i by construction, so A V = U S only confirms the code)
    print("3. Sorted and paired")
    print("   sigma decreasing:", bool(np.all(np.diff(sig) <= 0)))
    print(f"   max |A V - U S| = {np.abs(A @ V - U * sig).max():.2e}")
    print(f"   max |U^T U - I| = {np.abs(U.T @ U - np.eye(k_max)).max():.2e}")
    if k_max == min(m, n):
        print(f"   ||A - U S V^T||_F = {np.linalg.norm(A - (U * sig) @ V.T):.2e}")

    # 4. Accuracy through A^T A: relative error of each sigma_i against the library
    rel = np.abs(sig - s_lib[:k_max]) / s_lib[:k_max]
    cond = s_lib[0] / s_lib[-1]          # sigma_1 / sigma_min over the min(m, n) values
    print("4. Accuracy through A^T A")
    if m >= n:
        print(f"   cond(A) = {cond:.3e},  cond(A^T A) = cond(A)^2 = {cond ** 2:.3e}")
    else:
        print(f"   A^T A is {n} x {n} with rank <= {m}, so it is singular (cond = inf).")
        print(f"   On its nonzero part: (sigma_1 / sigma_{m})^2 = {cond ** 2:.3e}")
    print(f"   rel. error of sigma_1 = {rel[0]:.1e}, of sigma_{k_max} = {rel[-1]:.1e}, "
          f"max = {rel.max():.1e}")

    return sig, s_lib, rel


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Simulation first (low rank by construction)
    A_sim = simulate_low_rank(m=120, n=80, r=5, noise=0.01, seed=0)
    sig_s, lib_s, rel_s = check("Simulation", A_sim, k_max=80, ks=[5, 10])
    _, U5, _ = my_svd_power(A_sim, k=5)
    print(f"   Top 5 only: max |U^T U - I| = {np.abs(U5.T @ U5 - np.eye(5)).max():.2e}")

    # 2. Real image
    A_real = load_image(ROOT / "data" / "photo.jpeg")
    sig_r, lib_r, rel_r = check("Real image", A_real, k_max=51, ks=[5, 20, 50])

    # 3. Figure: singular values (log scale) and relative error of each sigma_i
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))

    axes[0].semilogy(np.arange(1, len(lib_s) + 1), lib_s, "o", ms=3, label="simulation")
    axes[0].semilogy(np.arange(1, 81), lib_r[:80], "s", ms=3, label="real image")
    axes[0].axvline(5, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("i")
    axes[0].set_ylabel(r"$\sigma_i$")
    axes[0].set_title("Assumption 1: singular values")
    axes[0].legend()

    axes[1].semilogy(np.arange(1, len(rel_s) + 1), np.maximum(rel_s, 1e-17), "o", ms=3, label="simulation")
    axes[1].semilogy(np.arange(1, len(rel_r) + 1), np.maximum(rel_r, 1e-17), "s", ms=3, label="real image")
    axes[1].set_xlabel("i")
    axes[1].set_ylabel(r"$|\sigma_i - \sigma_i^{lib}| / \sigma_i^{lib}$")
    axes[1].set_title("Assumption 4: accuracy through $A^TA$")
    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "assumption_checks.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\n=== Figures Saved ===")
    print(OUTPUT_DIR / "assumption_checks.png")


if __name__ == "__main__":
    main()
