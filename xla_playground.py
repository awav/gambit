from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)

@tf.function(experimental_compile=True)
def computation(a: Tensor, b: Tensor, tri_mat: Tensor) -> Tensor:
    outer = a @ tf.transpose(b)
    solve = tf.linalg.triangular_solve(tri_mat, outer)
    return solve @ tf.transpose(solve)
    #return tf.einsum("ij,kj->ik", solve, solve)

large_size = 4073*4073
a = tf.random.normal((1000, 10))
b = tf.random.normal((large_size, 10))
tri_mat = tf.random.uniform((1000, 1000))
# c = tf.random.normal((large_size, 1))

computation(a, b, tri_mat)
