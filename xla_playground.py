from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)

@tf.function(experimental_compile=True)
def computation(a: Tensor, b: Tensor, tri_mat: Tensor, c: Tensor) -> Tensor:
    outer = a @ tf.transpose(b)
    solve = tf.linalg.triangular_solve(tri_mat, outer)
    return solve @ c

large_size = 100000
a = tf.random.normal((1000, 10))
b = tf.random.normal((large_size, 10))
tri_mat = tf.random.uniform((1000, 1000))
c = tf.random.normal((large_size, 1))

computation(a, b, tri_mat, c)

# print(diff.numpy())

# @tf.function(experimental_compile=True)
# def outer_inner_add(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
#     ab = a @ b
#     return tf.math.exp(ab) @ c

# a = tf.random.normal((997, 10))
# b = tf.random.normal((10, 100000))
# c = tf.random.normal((100000, 1))
# outer_inner_add(a, b, c)
