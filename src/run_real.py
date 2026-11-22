"""Run the from-scratch SVD on the real image and verify it (Q5).

Cleaning steps (record these in the report):
  1. Load data/photo.jpeg (724 x 1086, RGB, uint8). No pixels are removed.
  2. Convert to grayscale: 0.299 R + 0.587 G + 0.114 B   (load_image).
  3. Convert to float and divide by 255, so every a_ij is in [0, 1].

Run:  python src/run_real.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from image_analysis import load_image
from methods import my_svd_power, my_rank_k, my_energy, my_storage_cost
from verify import report, align_signs

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "figures"
K_MAX = 50              # largest k we need
K_SHOW = [5, 20, 50]


# Smallest k with energy(k) >= p
def k_for(energy, p):
    """Input: energy (from my_energy), p (e.g. 0.95). Output: k (int).

    Stops with a message if the top K_MAX singular values keep less than p.
    """
    hit = np.flatnonzero(energy >= p)
    if hit.size == 0:
        raise SystemExit(f"The top {len(energy)} singular values keep less than {p:.0%}; raise K_MAX.")
    k = int(hit[0] + 1)
    return k


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the grayscale matrix A
    A = load_image(ROOT / "data" / "photo.jpeg")
    m, n = A.shape
    total = np.sum(A ** 2)          # ||A||_F^2 = sum of all sigma_i^2   (Iteration 2, Q4)

    print("\n=== Real Data A ===")
    print(f"Shape (m x n): {m} x {n}")
    print(f"Min={A.min():.4f}, Max={A.max():.4f}, Missing values (NaN): {np.isnan(A).sum()}")
    print(f"||A||_F^2 = {total:.4f}")

    # 2. From-scratch SVD, top K_MAX terms
    sig, U, V = my_svd_power(A, k=K_MAX)

    # 3. Library comparison (check only)
    U_lib, s_lib, Vt_lib = np.linalg.svd(A, full_matrices=False)
    sgn = align_signs(V, Vt_lib[:K_MAX].T)

    print("\n=== Ours vs numpy.linalg.svd (top 50) ===")
    report("sigma_1..50 vs library", sig, s_lib[:K_MAX])
    report("V (signs aligned)", V * sgn, Vt_lib[:K_MAX].T)
    report("U (signs aligned)", U * sgn, U_lib[:, :K_MAX])

    for k in K_SHOW:
        A_lib = (U_lib[:, :k] * s_lib[:k]) @ Vt_lib[:k]       # A_k has no sign issue
        report(f"A_{k} vs library", my_rank_k(sig, U, V, k), A_lib)

    # 4. Q5: energy(k) = sum_{i<=k} sigma_i^2 / ||A||_F^2
    energy = my_energy(sig, A)
    k95 = k_for(energy, 0.95)
    k99 = k_for(energy, 0.99)

    # Same rule on the centred image (mean brightness removed), for comparison only
    A_centred = A - A.mean()
    sig_c, _, _ = my_svd_power(A_centred, k=K_MAX)
    energy_c = my_energy(sig_c, A_centred)
    k95_c = k_for(energy_c, 0.95)

    print("\n=== Q5. Energy ===")
    print(f"sigma_1^2 alone keeps {energy[0]:.2%} of the energy")
    print(f"Mean brightness part: mn * mean(A)^2 / ||A||_F^2 = {m * n * A.mean() ** 2 / total:.2%}")
    print(f"k for 95% energy = {k95}")
    print(f"k for 99% energy = {k99}")
    print(f"Centred image: k for 95% energy = {k95_c}")

    # 5. Error and storage for each k
    #    Check: ||A - A_k||_F^2 = ||A||_F^2 - sum_{i<=k} sigma_i^2    (Iteration 2, Q5)
    print("\n=== Error and Storage Cost ===")
    print(f"{'k':>4s} {'energy':>8s} {'||A-A_k||_F^2':>14s} {'tail sum':>10s} "
          f"{'rel. error':>10s} {'storage':>8s} {'storage/mn':>10s}")

    ks = sorted(set([k95, k99] + K_SHOW))
    approximations = {}
    for k in ks:
        Ak = my_rank_k(sig, U, V, k)
        error = np.sum((A - Ak) ** 2)
        tail = total - np.sum(sig[:k] ** 2)
        storage = my_storage_cost(k, m, n)
        approximations[k] = Ak
        print(f"{k:4d} {energy[k - 1]:8.4f} {error:14.6f} {tail:10.6f} "
              f"{np.sqrt(error / total):10.4f} {storage:8d} {storage / (m * n):10.4f}")
    print(f"Original image: mn = {m * n} numbers")

    # 6. Figure 1: energy curve
    k_axis = np.arange(1, K_MAX + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(k_axis, energy, "o-", ms=3, label="original image")
    plt.plot(k_axis, energy_c, "s-", ms=3, label="centred image")
    plt.axhline(0.95, color="gray", ls="--", lw=1)
    plt.axvline(k95, color="C0", ls=":", lw=1)
    plt.axvline(k95_c, color="C1", ls=":", lw=1)
    plt.xlabel("k")
    plt.ylabel(r"energy $\sum_{i\leq k}\sigma_i^2 / \|A\|_F^2$")
    plt.title("Energy kept by the top k singular values")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "real_energy.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 7. Figure 2: reconstructions (clipped to [0, 1] for display only)
    shown = [k95] + K_SHOW
    fig, axes = plt.subplots(1, len(shown) + 1, figsize=(3.2 * (len(shown) + 1), 2.6))

    axes[0].imshow(A, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original\n{m * n} numbers", fontsize=9)
    axes[0].axis("off")

    for ax, k in zip(axes[1:], shown):
        ax.imshow(np.clip(approximations[k], 0, 1), cmap="gray", vmin=0, vmax=1)
        label = " (95% energy)" if k == k95 else ""
        ax.set_title(f"k = {k}{label}\n{my_storage_cost(k, m, n)} numbers", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "real_reconstructions.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print("\n=== Figures Saved ===")
    print(OUTPUT_DIR / "real_energy.png")
    print(OUTPUT_DIR / "real_reconstructions.png")


if __name__ == "__main__":
    main()
