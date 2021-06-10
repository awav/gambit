from collections import namedtuple
from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)
State = namedtuple("State", "i, outer, sum")

@tf.function(experimental_compile=True)
def run(large_size, N=10):
  x = tf.random.normal((large_size, 100))
  y = tf.random.normal((large_size, 100))
  v = tf.random.normal((large_size, 5))

  outer = x @ tf.transpose(y)

  size = large_size // N
  slices = [tf.slice(outer, [size * n, 0], [size, large_size]) for n in range(0, N)]

  def cond(state):
    return state.i < 10

  def body(state):
    sum_next = state.sum + tf.concat([o @ v for o in state.outer], 0)
    i_next = state.i + 1
    return [State(i=i_next, outer=state.outer, sum=sum_next)]

  state_0 = State(i=0, outer=tuple(slices), sum=tf.fill((large_size, 5), 0.0))
  [states] = tf.while_loop(cond, body, [state_0])
  return states.sum

run(large_size=2_000)
