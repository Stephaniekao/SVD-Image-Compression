"""Quick tests for src/methods.py, using the 2 x 2 example A = [[3, 0], [4, 5]].

Run:  python src/test_methods.py
Write one function, then run this file and look at that function's line.
(Lines for functions you have not written yet will stop with NotImplementedError;
comment them out for now.)

Signs: a singular vector v and -v are both correct, so U and V may differ
from the expected values below by a sign in a whole column.
"""
import numpy as np
from methods import (my_norm, my_gram, my_power_iteration, my_deflate,
                     my_svd_power, my_rank_k, my_energy, my_storage_cost)

np.set_printoptions(precision=4, suppress=True)
A = np.array([[3.0, 0.0], [4.0, 5.0]])

# 1. my_norm                       expected: 5.0
print("1. my_norm:", my_norm(np.array([3.0, 4.0])))

# 2. my_gram                       expected: [[25. 20.] [20. 25.]]
M = my_gram(A)
print("2. my_gram:\n", M)

# 3. my_power_iteration            expected: lam = 45.0, v = [0.7071 0.7071] (or both negative)
lam, v = my_power_iteration(M)
print("3. my_power_iteration: lam =", lam, " v =", v)

# 4. my_deflate                    expected: [[ 2.5 -2.5] [-2.5  2.5]]
print("4. my_deflate:\n", my_deflate(M, lam, v))

# 5. my_svd_power                  expected: sig = [6.7082 2.2361]  (= 3*sqrt(5), sqrt(5))
#                                  U columns: +-[0.3162 0.9487], +-[0.9487 -0.3162]
#                                  V columns: +-[0.7071 0.7071], +-[0.7071 -0.7071]
sig, U, V = my_svd_power(A, k=2)
print("5. my_svd_power: sig =", sig, " exact:", np.array([3 * np.sqrt(5), np.sqrt(5)]))
print("   U =\n", U)
print("   V =\n", V)
print("   U^T U (expected identity):\n", U.T @ U)
print("   A v_i - sigma_i u_i (expected all 0):\n", A @ V - U * sig)

# 6. my_rank_k                     expected k=1: [[1.5 1.5] [4.5 4.5]]   k=2: A itself
A1 = my_rank_k(sig, U, V, 1)
print("6. my_rank_k, k=1:\n", A1)
print("   k=2 (expected A):\n", my_rank_k(sig, U, V, 2))
print("   ||A - A_1||_F^2 =", np.sum((A - A1) ** 2), " expected sigma_2^2 = 5.0")

# 7. my_energy                     expected: [0.9 1. ]   (45/50, 50/50)
print("7. my_energy:", my_energy(sig, A))

# 8. my_storage_cost               expected: 5   (k=1, m=n=2: 1*(2+2+1))
print("8. my_storage_cost:", my_storage_cost(1, 2, 2))
print("   k=5, m=120, n=80:", my_storage_cost(5, 120, 80), " expected 1005")