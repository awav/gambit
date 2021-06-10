from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)

# $exp(||X-Y||^2) v$ (where exp is pointwise)
@tf.function(experimental_compile=True)
def run(x: Tensor, y: Tensor, v: Tensor) -> Tensor:
    xx = tf.reduce_sum(tf.square(x), 1)
    yy = tf.reduce_sum(tf.square(y), 1)

    xx = tf.reshape(xx, (-1, 1))
    yy = tf.reshape(yy, (1, -1))

    D = tf.sqrt(tf.maximum(xx + yy - 2.0 * tf.matmul(x, y, False, True), 0.0))

    e = tf.math.exp(D)
    return e @ v

arguments = [
  (1000, 2), (1000, 2), (1000, 1)
]
