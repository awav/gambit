import tensorflow as tf
import gpflow

Matrix = tf.Tensor
Vector = tf.Tensor


def triangular_solve(
    tril_matrix: Matrix, kernel: gpflow.kernels.Kernel, x: Vector, y: Vector
) -> Vector:
    k_xy = kernel(x, y)
    solution = tf.linalg.triangular_solve(tril_matrix, k_xy)
    return solution
