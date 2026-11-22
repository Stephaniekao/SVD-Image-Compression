"""Small worked example: SVD of A = [[3, 0], [4, 5]] step by step (Q1, Q2).

Exact values use fractions.Fraction. A square root such as sqrt(5) is not a
fraction, so each vector is kept as (integer vector) / sqrt(integer). Everything
that only needs squares (lambda_i, sigma_i^2, v_i v_i^T, A_1, ||A - A_1||_F^2)
stays exact.
At the end, the same matrix goes through our my_svd_power, and we compare it
with the exact values and with numpy.linalg.svd (check only).

Run:  python src/small_example.py
"""
from fractions import Fraction
from math import gcd, isqrt

import numpy as np

from methods import my_svd_power, my_rank_k
from verify import report, align_signs


def show(M):
    """Print a 2 x 2 matrix of Fractions. Input: M (list of 2 rows)."""
    for row in M:
        print("    [" + ", ".join(f"{str(x):>5s}" for x in row) + "]")


def matmul(X, Y):
    """Exact matrix product. Input: X (p x q), Y (q x r) as lists. Output: X Y (p x r)."""
    p, q, r = len(X), len(Y), len(Y[0])
    Z = []
    for i in range(p):
        row = []
        for j in range(r):
            row.append(sum(X[i][t] * Y[t][j] for t in range(q)))
        Z.append(row)
    return Z


def transpose(X):
    """Input: X (p x q) as a list. Output: X^T (q x p)."""
    return [list(col) for col in zip(*X)]


# v v^T = w w^T / ||w||^2  when v = w / ||w||
def outer_unit(w):
    """Exact v v^T for v = w / ||w||. Input: w (integer vector). Output: 2 x 2 matrix."""
    length_sq = sum(x * x for x in w)
    P = []
    for a in w:
        P.append([Fraction(a * b, length_sq) for b in w])
    return P


def simplify(w):
    """Divide an integer vector by its gcd, with a positive first entry.

    Input:  w (integer vector).
    Output: w (simplified), and the text "(w) / sqrt(||w||^2)" of its unit vector.
    """
    g = abs(gcd(*w))
    if w[0] < 0 or (w[0] == 0 and w[1] < 0):
        g = -g
    w = [x // g for x in w]
    text = f"({w[0]}, {w[1]}) / sqrt({w[0] ** 2 + w[1] ** 2})"
    return w, text


A = [[Fraction(3), Fraction(0)],
     [Fraction(4), Fraction(5)]]

print("\n=== A ===")
show(A)

# Step 1:  M = A^T A                                     (Iteration 2, Q1)
M = matmul(transpose(A), A)

print("\n=== Step 1. M = A^T A ===")
show(M)
print("Symmetric (M^T = M):", transpose(M) == M)

# Step 2:  det(M - lambda I) = lambda^2 - tr(M) lambda + det(M) = 0
tr = M[0][0] + M[1][1]
det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
disc = tr * tr - 4 * det
root = isqrt(int(disc))          # 1600 = 40^2, so both roots are exact
lam1 = (tr + root) / 2
lam2 = (tr - root) / 2

print("\n=== Step 2. Characteristic polynomial ===")
print(f"lambda^2 - {tr} lambda + {det} = 0")
print(f"Discriminant = {tr}^2 - 4 * {det} = {disc} = {root}^2:", root * root == disc)
print(f"lambda_1 = ({tr} + {root}) / 2 = {lam1}")
print(f"lambda_2 = ({tr} - {root}) / 2 = {lam2}")
print("Both >= 0 (Iteration 2, Q1):", lam1 >= 0 and lam2 >= 0)

# Step 3:  sigma_i = sqrt(lambda_i)
print("\n=== Step 3. Singular values ===")
print(f"sigma_1^2 = {lam1},  sigma_1 = sqrt(45) = 3 sqrt(5) = {np.sqrt(45):.6f}")
print(f"sigma_2^2 = {lam2},   sigma_2 = sqrt(5) = {np.sqrt(5):.6f}")

# Step 4:  (M - lambda I) w = 0,  v = w / ||w||
print("\n=== Step 4. Eigenvectors ===")
w_list = []
for lam in [lam1, lam2]:
    B = [[M[0][0] - lam, M[0][1]],
         [M[1][0], M[1][1] - lam]]
    w = [int(-B[0][1]), int(B[0][0])]       # (b, -a) solves a x + b y = 0
    w, text = simplify(w)
    w_list.append(w)
    print(f"lambda = {lam}:  M - lambda I =")
    show(B)
    print(f"  v = {text}")

# Step 5:  u_i = A v_i / sigma_i                          (Iteration 2, Q2)
print("\n=== Step 5. u_i = A v_i / sigma_i ===")
u_list = []
for i, w in enumerate(w_list):
    Aw = [int(A[0][0] * w[0] + A[0][1] * w[1]),
          int(A[1][0] * w[0] + A[1][1] * w[1])]
    z, text = simplify(Aw)
    u_list.append(z)
    print(f"A w_{i + 1} = ({Aw[0]}, {Aw[1]})  ->  u_{i + 1} = {text}")

# Second-way check:  M w = lambda w,  v_1^T v_2 = 0,  u_1^T u_2 = 0
print("\n=== Second-way checks (exact) ===")
for w, lam in zip(w_list, [lam1, lam2]):
    Mw = [M[0][0] * w[0] + M[0][1] * w[1],
          M[1][0] * w[0] + M[1][1] * w[1]]
    print(f"M ({w[0]}, {w[1]}) = ({Mw[0]}, {Mw[1]}) = {lam} * ({w[0]}, {w[1]}):",
          Mw == [lam * w[0], lam * w[1]])
print("v_1^T v_2 =", w_list[0][0] * w_list[1][0] + w_list[0][1] * w_list[1][1])
print("u_1^T u_2 =", u_list[0][0] * u_list[1][0] + u_list[0][1] * u_list[1][1])

# sigma_i u_i v_i^T = A v_i v_i^T,  so the sum is A (v_1 v_1^T + v_2 v_2^T) = A   (Iteration 2, Q3)
P1 = outer_unit(w_list[0])
P2 = outer_unit(w_list[1])
term1 = matmul(A, P1)
term2 = matmul(A, P2)
total = [[term1[i][j] + term2[i][j] for j in range(2)] for i in range(2)]

print("v_1 v_1^T =")
show(P1)
print("v_2 v_2^T =")
show(P2)
print("sigma_1 u_1 v_1^T + sigma_2 u_2 v_2^T = A P_1 + A P_2 =")
show(total)
print("Equals A:", total == A)

# Q2:  A_1 = A v_1 v_1^T,  ||A - A_1||_F^2 = sigma_2^2   (Iteration 2, Q5)
A1 = term1
R = [[A[i][j] - A1[i][j] for j in range(2)] for i in range(2)]
squares = [x * x for row in R for x in row]
err = sum(squares)

print("\n=== Q2. Rank-1 approximation ===")
print("A_1 = sigma_1 u_1 v_1^T = A v_1 v_1^T =")
show(A1)
print("A - A_1 =")
show(R)
print(f"||A - A_1||_F^2 = {' + '.join(str(x) for x in squares)} = {err}")
print(f"sigma_2^2 = {lam2}.  Equal:", err == lam2)
print(f"||A||_F^2 = {sum(x * x for row in A for x in row)},  "
      f"sigma_1^2 + sigma_2^2 = {lam1 + lam2}")

# Same matrix through our from-scratch code, then numpy (check only)
print("\n=== my_svd_power vs exact values and numpy.linalg.svd ===")
A_np = np.array([[3.0, 0.0], [4.0, 5.0]])
sig, U, V = my_svd_power(A_np, k=2)

sig_exact = np.array([3 * np.sqrt(5), np.sqrt(5)])
V_exact = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
U_exact = np.array([[1, 3], [3, -1]]) / np.sqrt(10)
sgn = align_signs(V, V_exact)
report("sigma vs exact", sig, sig_exact)
report("V vs exact (signs aligned)", V * sgn, V_exact)
report("U vs exact (signs aligned)", U * sgn, U_exact)

U_lib, s_lib, Vt_lib = np.linalg.svd(A_np)
sgn = align_signs(V, Vt_lib.T)
report("sigma vs numpy", sig, s_lib)
report("V vs numpy (signs aligned)", V * sgn, Vt_lib.T)
report("U vs numpy (signs aligned)", U * sgn, U_lib)

A1_np = my_rank_k(sig, U, V, 1)
report("A_1 vs exact", A1_np, np.array([[1.5, 1.5], [4.5, 4.5]]))
print(f"||A - A_1||_F^2 = {np.sum((A_np - A1_np) ** 2):.12f}  (exact 5)")
