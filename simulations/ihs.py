from math import sqrt
import numpy as np
import numpy.random as npr
import numpy.linalg as npla
import scipy as sp
import scipy.linalg as spla

def sketch_matrix(m, n, type="gaussian"):
    if type == "gaussian":
        return npr.normal(size=(m, n)) / np.sqrt(m)
    elif type == "exponential":
        return npr.laplace(scale=sqrt(0.5), size=(m, n)) / np.sqrt(m)
    elif type == "subsample":
        S = np.zeros((m, n))
        S[range(m), npr.choice(n, m, replace=False)] = np.sqrt(n * 1.0 / m)
        return S
    elif type == "orthogonal":
        return spla.svd(npr.normal(size=(n, m)), full_matrices=False,
                compute_uv=True, check_finite=False)[0].T * np.sqrt(n * 1.0 / m)
    else:
        raise NotImplementedError("<type> not identified.")

def ols(y, A):
    return spla.lstsq(A, y, check_finite=False, lapack_driver="gelsy")[0]

def classical_sketch(y, A, m, sketch_type="gaussian"):
    n, p = A.shape
    S = sketch_matrix(m, n, type=sketch_type)
    return ols(np.dot(S, y), np.dot(S, A))

def hessian_sketch(y, A, m, sketch_type="gaussian"):
    n, p = A.shape
    S = sketch_matrix(m, n, type=sketch_type) # S^TS should be cloesd to I.
    SA = np.dot(S, A)
    return spla.solve(np.dot(SA.T, SA), np.dot(A.T, y), check_finite=False, assume_a="pos")

def iterative_hessian_sketch(y, A, m, iter_num, sketch_type="gaussian"):
    n, p = A.shape
    x = np.zeros(p, dtype=np.float64)
    for i in range(iter_num):
        x += hessian_sketch(y - np.dot(A, x), A, m, sketch_type=sketch_type)
    return x

def pred_error(x1, x2, A):
    return npla.norm(np.dot(A, x1 - x2), 2) / sqrt(A.shape[0])

def mse_error(x1, x2):
    return npla.norm(x1 - x2, 2) / sqrt(x1.shape[0])
 
