from collections import namedtuple
from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)
State = namedtuple("State", "i, sum")

@tf.function(experimental_compile=True)
def xla(x: Tensor, y: Tensor):
  diff = x[None, :, :] - y[:, None, :]
  squares = diff ** 2
  dists = tf.reduce_sum(squares, -1)
  return dists

def no_xla(x: Tensor, y: Tensor):
  diff = x[None, :, :] - y[:, None, :]
  squares = diff ** 2
  dists = tf.reduce_sum(squares, -1)
  return dists

large_size = 2_000
x = tf.random.normal((large_size, 100))
y = tf.random.normal((large_size, 100))
v = tf.random.normal((large_size, 5))

a = xla(x, y)
b = no_xla(x, y)

print(a)
print(b)
print(a - b)
print(tf.math.reduce_max(a - b))
