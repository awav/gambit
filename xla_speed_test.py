from typing import TypeVar
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

Tensor = TypeVar("Tensor", bound=tf.Tensor)

# $exp(||X-Y||^2) v$ (where exp is pointwise)
@tf.function(experimental_compile=True)
def test_dist_matrix(x: Tensor, y: Tensor, v: Tensor) -> Tensor:
    xx = tf.reduce_sum(tf.square(x), 1)
    yy = tf.reduce_sum(tf.square(y), 1)

    xx = tf.reshape(xx, (-1, 1))
    yy = tf.reshape(yy, (1, -1))

    D = tf.sqrt(tf.maximum(xx + yy - 2.0 * tf.matmul(x, y, False, True), 0.0))

    e = tf.math.exp(D)
    return e @ v

def run(large_size):
  x = tf.random.normal((large_size, 100))
  y = tf.random.normal((large_size, 100))
  v = tf.random.normal((large_size, 5))
  for _ in range(1000):
    res = test_dist_matrix(x, y, v)

run(large_size=50_000)
