import numpy as np

rng = np.random.default_rng(0)
m, n, r = 120, 80, 5

A = (
    rng.normal(size=(m, r))
    @ rng.normal(size=(r, n))
    + 0.01 * rng.normal(size=(m, n))
)

print("Matrix shape:", A.shape)
print("Matrix rank:", np.linalg.matrix_rank(A))

# Q1: Eigen-decomposition of A.T @ A
lam, V = np.linalg.eigh(A.T @ A)

# Sort eigenvalues and eigenvectors in descending order
lam = lam[::-1]
V = V[:, ::-1]

# Compute singular values
sig = np.sqrt(np.clip(lam, 0, None))

print("\nFirst 10 singular values:")
print(np.round(sig[:10], 4))

print("\nMinimum eigenvalue:", lam.min())
print("All eigenvalues non-negative (within tolerance):",
      np.all(lam >= -1e-10))


# Q2: Construct the first r left singular vectors
U = A @ V[:, :r] / sig[:r]

# Check orthonormality
orthogonality_error = np.max(
    np.abs(U.T @ U - np.eye(r))
)

print("\nQ2: Orthonormality Check")
print("U shape:", U.shape)
print("U.T @ U shape:", (U.T @ U).shape)
print("Maximum error:", orthogonality_error)


# Q3: Reconstruct A using the sum of rank-one matrices
# The noisy matrix has full rank, so use all nonzero singular values
rank_A = np.linalg.matrix_rank(A)

U_full = A @ V[:, :rank_A] / sig[:rank_A]

A_reconstructed = np.zeros_like(A)

for i in range(rank_A):
    A_reconstructed += sig[i] * np.outer(
        U_full[:, i], V[:, i]
    )

q3_error = np.linalg.norm(A - A_reconstructed, ord="fro")

print("\nQ3: Rank-One Reconstruction Check")
print("Number of components:", rank_A)
print("Reconstruction error:", q3_error)

# Compare the known underlying rank with the observed singular values
print("\nSimulation: True vs. Observed")
print("True underlying rank:", r)
print("Expected drop after component:", r)
print("Observed fifth singular value:", sig[4])
print("Observed sixth singular value:", sig[5])
print("Ratio of fifth to sixth:", sig[4] / sig[5])


# Q4: Frobenius norm identity
frobenius_squared = np.sum(A**2)
trace_value = np.trace(A.T @ A)
singular_squared = np.sum(sig**2)

print("\nQ4: Frobenius Norm Check")
print("Frobenius norm squared:", frobenius_squared)
print("Trace of A.T @ A:", trace_value)
print("Sum of singular values squared:", singular_squared)

print(
    "Maximum difference:",
    max(
        abs(frobenius_squared - trace_value),
        abs(frobenius_squared - singular_squared)
    )
)


# Q5: Rank-k approximation and storage cost
print("\nQ5: Rank-k Approximation Check")

for k in [1, 3, 5, 10]:
    Ak = A @ V[:, :k] @ V[:, :k].T

    reconstruction_error = np.sum((A - Ak)**2)
    theoretical_error = np.sum(sig[k:]**2)
    storage_cost = k * (m + n + 1)

    print(f"\nk = {k}")
    print("Reconstruction error:", reconstruction_error)
    print("Theoretical error:", theoretical_error)
    print("Difference:", abs(reconstruction_error - theoretical_error))
    print("Storage cost:", storage_cost)

# Final check: Compare with NumPy SVD
sig_numpy = np.linalg.svd(A, compute_uv=False)

print("\nFinal Check: NumPy SVD Comparison")
print("Our first 5 singular values:", np.round(sig[:5], 6))
print("NumPy first 5 singular values:", np.round(sig_numpy[:5], 6))

print(
    "Maximum singular value difference:",
    np.max(np.abs(sig - sig_numpy))
)
