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
    norm = np.sqrt(x @ x)
    return norm


# Step 1:  M = A^T A            (Iteration 2, Q1: M is symmetric PSD)
def my_gram(A):
    """Input: A (m x n). Output: M (n x n)."""
    M = np.transpose(A) @ A
    return M


# Step 2:  v_{t+1} = M v_t / ||M v_t||,   lambda = v^T M v
def my_power_iteration(M, tol=1e-12, max_iter=10_000, seed=0):
    """Largest eigenpair of a symmetric PSD matrix M.

    Input:  M (n x n).
    Output: lam (float), v (length n, ||v|| = 1).
    Stop when ||v_{t+1} - v_t|| < tol.
    """
    n = M.shape[0]
    rng = np.random.default_rng(seed)
    v = rng.normal(size=n)
    v /= my_norm(v)

    for i in range(max_iter):
        w = (M @ v) 
        w /= my_norm(w)
        diff = my_norm(w - v)
        v = w
        if diff < tol:
            break
        
    lam = v @ M @ v
    return lam, v


# Step 3:  M <- M - lambda v v^T          (deflation)
def my_deflate(M, lam, v):
    """Remove the found eigenpair. Input: M (n x n), lam, v. Output: new M (n x n)."""
    return M - lam * np.outer(v, v)


# Step 4:  sigma_i = sqrt(lambda_i),  u_i = A v_i / sigma_i   (Iteration 2, Q2)
def my_svd_power(A, k, tol=1e-12, max_iter=10_000, seed=0):
    """Top-k SVD by power iteration + deflation (repeat Steps 2-3 k times).

    Input:  A (m x n), k.
    Output: sig (length k, largest first),
            U   (m x k, columns u_i),
            V   (n x k, columns v_i)   <- V, not V^T
    """
    m, n = A.shape
    sig = np.zeros(k)
    U = np.zeros((m, k))
    V = np.zeros((n, k))
    M = my_gram(A)

    for i in range(k):
        if not M.any():
            break
        lam, v = my_power_iteration(M, tol=tol, max_iter=max_iter, seed=seed)
        if lam <= 0:
            break
        sig[i] = np.sqrt(lam)
        V[:, i] = v
        U[:, i] = (A @ v) / sig[i]
        if i + 1 < k:
           M = my_deflate(M, lam, v)

    return sig, U, V


# Step 5:  A_k = sum_{i<=k} sigma_i u_i v_i^T = U_k Sigma_k V_k^T   (Iteration 2, Q3)
def my_rank_k(sig, U, V, k):
    """Input: sig, U, V from my_svd_power, k. Output: A_k (m x n)."""
    return (U[:, :k] * sig[:k]) @ V[:, :k].T

# Step 6:  energy(k) = sum_{i<=k} sigma_i^2 / ||A||_F^2      (Iteration 2, Q4)
def my_energy(sig, A):
    """Cumulative energy for k = 1, ..., len(sig).

    Denominator ||A||_F^2 = sum of all sigma_i^2, so only the top sigmas are needed.
    Input: sig (length k), A (m x n). Output: array (length k).
    """
    A = np.asarray(A, dtype=float)
    sig = np.asarray(sig, dtype=float)
    total = np.sum(A**2)
    cumulative = np.cumsum(sig**2)

    if total == 0:
        return np.zeros_like(cumulative)
    
    cumulative /= total
    return cumulative

# Step 7:  storage = k (m + n + 1)                            (Iteration 2, Q5)
def my_storage_cost(k, m, n):
    """Numbers stored for a rank-k approximation. Output: int."""
    return k * (m + n + 1)


