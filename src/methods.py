"""Project 3 - SVD from scratch (NumPy only, no numpy.linalg).

Each function is one step of the Iteration 2 algorithm.
Symbols follow the Overleaf report: A (m x n), M = A^T A (n x n),
lambda_i, sigma_i = sqrt(lambda_i), u_i (length m), v_i (length n).

The same functions run the small example, the simulation and the real data.
"""
import numpy as np


# Helper:  ||x||_2 = sqrt(x^T x)
def my_norm(x):
    """Euclidean norm of a vector. Input: x (length p). Output: float."""
    raise NotImplementedError   # TODO


# Step 1:  M = A^T A            (Iteration 2, Q1: M is symmetric PSD)
def my_gram(A):
    """Input: A (m x n). Output: M (n x n)."""
    raise NotImplementedError   # TODO


# Step 2:  v_{t+1} = M v_t / ||M v_t||,   lambda = v^T M v
def my_power_iteration(M, tol=1e-12, max_iter=10_000, seed=0):
    """Largest eigenpair of a symmetric PSD matrix M.

    Input:  M (n x n).
    Output: lam (float), v (length n, ||v|| = 1).
    Stop when ||v_{t+1} - v_t|| < tol.
    """
    raise NotImplementedError   # TODO


# Step 3:  M <- M - lambda v v^T          (deflation)
def my_deflate(M, lam, v):
    """Remove the found eigenpair. Input: M (n x n), lam, v. Output: new M (n x n)."""
    raise NotImplementedError   # TODO


# Step 4:  sigma_i = sqrt(lambda_i),  u_i = A v_i / sigma_i   (Iteration 2, Q2)
def my_svd_power(A, k, tol=1e-12, max_iter=10_000, seed=0):
    """Top-k SVD by power iteration + deflation (repeat Steps 2-3 k times).

    Input:  A (m x n), k.
    Output: sig (length k, largest first),
            U   (m x k, columns u_i),
            V   (n x k, columns v_i)   <- V, not V^T
    """
    raise NotImplementedError   # TODO


# Step 5:  A_k = sum_{i<=k} sigma_i u_i v_i^T = U_k Sigma_k V_k^T   (Iteration 2, Q3)
def my_rank_k(sig, U, V, k):
    """Input: sig, U, V from my_svd_power, k. Output: A_k (m x n)."""
    raise NotImplementedError   # TODO


# Step 6:  energy(k) = sum_{i<=k} sigma_i^2 / ||A||_F^2      (Iteration 2, Q4)
def my_energy(sig, A):
    """Cumulative energy for k = 1, ..., len(sig).

    Denominator ||A||_F^2 = sum of all sigma_i^2, so only the top sigmas are needed.
    Input: sig (length k), A (m x n). Output: array (length k).
    """
    raise NotImplementedError   # TODO


# Step 7:  storage = k (m + n + 1)                            (Iteration 2, Q5)
def my_storage_cost(k, m, n):
    """Numbers stored for a rank-k approximation. Output: int."""
    raise NotImplementedError   # TODO
