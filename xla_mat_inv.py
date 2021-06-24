from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)

@tf.function(experimental_compile=True)
def test(A: Tensor, c: Tensor) -> Tensor:
    A_inv = tf.linalg.inv(A)
    prod = tf.tensordot(A_inv, c, [[1], [0]])
    return tf.tensordot(c, prod, [[0], [0]])

x = tf.random.normal((100, 100))
c = tf.random.normal((100,))
test(x, c)
