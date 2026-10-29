"""A tiny, dependency-free linear-algebra kernel for the Gaussian process.

Only what the GP needs: Cholesky factorization of a symmetric positive-definite
matrix and triangular solves. Pure Python lists of lists; sizes are the number of
trials, which stays small for hyperparameter search.
"""

from __future__ import annotations

from collections.abc import Sequence

Matrix = list[list[float]]
Vector = list[float]


def cholesky(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Return the lower-triangular Cholesky factor ``L`` with ``L @ L.T == matrix``."""
    n = len(matrix)
    lower: Matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                diagonal = matrix[i][i] - total
                if diagonal <= 0.0:
                    diagonal = 1e-12
                lower[i][j] = diagonal**0.5
            else:
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


def _forward_substitution(lower: Matrix, vector: Sequence[float]) -> Vector:
    n = len(lower)
    result: Vector = [0.0] * n
    for i in range(n):
        total = sum(lower[i][k] * result[k] for k in range(i))
        result[i] = (vector[i] - total) / lower[i][i]
    return result


def _back_substitution(upper: Matrix, vector: Sequence[float]) -> Vector:
    n = len(upper)
    result: Vector = [0.0] * n
    for i in reversed(range(n)):
        total = sum(upper[i][k] * result[k] for k in range(i + 1, n))
        result[i] = (vector[i] - total) / upper[i][i]
    return result


def cholesky_solve(lower: Matrix, vector: Sequence[float]) -> Vector:
    """Solve ``A x = vector`` given the Cholesky factor ``lower`` of ``A``."""
    y = _forward_substitution(lower, vector)
    upper = [[lower[j][i] for j in range(len(lower))] for i in range(len(lower))]
    return _back_substitution(upper, y)
