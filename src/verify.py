"""Shared check helpers (the report() helper is from the Iteration 3 handout).

Every comparison in small_example.py, check_simulation.py, run_real.py and
check_assumptions.py is printed in the same format.
"""
import numpy as np


def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||.

    Input:  name (str), mine and ref (arrays of the same shape).
    Output: abs_err, rel_err (floats).
    """
    mine = np.asarray(mine, float)
    ref = np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")
    return abs_err, rel_err


# Sign of column i:  sign(x_i^T y_i)      (v and -v are both unit eigenvectors)
def align_signs(X, Y):
    """Signs that make each column of X point the same way as the matching column of Y.

    Input:  X, Y (p x k).
    Output: sgn (length k, entries +1 or -1). Use X * sgn.
    """
    sgn = np.sign(np.sum(X * Y, axis=0))
    sgn[sgn == 0] = 1
    return sgn
