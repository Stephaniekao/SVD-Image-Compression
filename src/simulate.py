"""Project 3 simulation (moved from the Iteration 2 script simulation.py).

Model:  A = L + noise * E,   L = B C,
        B in R^{m x r}, C in R^{r x n}, E in R^{m x n}, all entries ~ N(0, 1).
L has rank r exactly (the truth); A is L plus small noise.
"""
import numpy as np


def simulate_low_rank(m=120, n=80, r=5, noise=0.01, seed=0, return_truth=False):
    """Return A (m x n). With return_truth=True, return (A, L), where L = B C is the noise-free rank-r matrix.

    The random draws are made in the same order as in Iteration 2
    (B, then C, then E), so seed=0 gives exactly the same A as last week.
    """
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(m, r))
    C = rng.normal(size=(r, n))
    L = B @ C                                  # truth: rank r
    A = L + noise * rng.normal(size=(m, n))    # observed data
    if return_truth:
        return A, L
    return A


if __name__ == "__main__":
    # Check: the function reproduces the Iteration 2 matrix exactly.
    rng = np.random.default_rng(0)
    A_old = (rng.normal(size=(120, 5)) @ rng.normal(size=(5, 80))
             + 0.01 * rng.normal(size=(120, 80)))
    A, L = simulate_low_rank(return_truth=True)

    print("A shape:", A.shape)
    print("Max |A - A_iteration2|:", np.abs(A - A_old).max())   # should be 0.0
    print("Max |A - L| (noise size):", np.abs(A - L).max())
