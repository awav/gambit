from collections import namedtuple
from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)
State = namedtuple("State", "i, sum")

@tf.function(experimental_compile=True)
def run(large_size):
  x = tf.random.normal((large_size, 100))
  y = tf.random.normal((large_size, 100))
  v = tf.random.normal((large_size, 5))

  outer = x @ tf.transpose(y)

  def cond(state):
    return state.i < 10

  def body(state):
    sum_next = state.sum + outer @ v
    i_next = state.i + 1
    return [State(i=i_next, sum=sum_next)]

  state_0 = State(i=0, sum=tf.fill((large_size, 5), 0.0))
  [states] = tf.while_loop(cond, body, [state_0], parallel_iterations=100)
  return states.sum

run(large_size=2_000)
