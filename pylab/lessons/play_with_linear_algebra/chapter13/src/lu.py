from src._globals import is_zero
from src.matrix import Matrix
from src.vector import Vector

def lu(matrix):
    assert matrix.row_num() == matrix.col_num(), "matrix must be a square matrix"

    n = matrix.row_num()
    A = [matrix.row_vector(i) for i in range(n)]

    L = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(n)]

    for i in range(n):
        # 看A[i][i]位置是否可以是主元
        if is_zero(A[i][i]):
            return None, None
        else:
            for j in range(i + 1, n):
                p = A[j][i] / A[i][i]
                A[j] = A[j] - p * A[i]
                L[j][i] = p

    return Matrix(L), Matrix([A[i].underlying_list() for i in range(n)])