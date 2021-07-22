from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)

@tf.function(experimental_compile=True)
def dots_lhs(a: Tensor, b: Tensor, c: Tensor, d: Tensor, e: Tensor) -> Tensor:
  ab = (a @ tf.transpose(b))
  abc = ab @ c
  abcd = abc @ tf.transpose(d)
  return abcd @ e

@tf.function(experimental_compile=True)
def dots_rhs(a: Tensor, b: Tensor, c: Tensor, d: Tensor, e: Tensor) -> Tensor:
  de = d @ tf.transpose(e) # N x N
  cde = tf.transpose(c) @ de
  bcde = b @ cde
  return tf.transpose(a) @ bcde

n, m = 1000, 2
a = tf.random.normal((n, m))
b = tf.random.normal((n, m))
c = tf.random.normal((n, m))
d = tf.random.normal((n, m))
e = tf.random.normal((n, m))

print(dots_rhs(a, b, c, d, e).numpy())
