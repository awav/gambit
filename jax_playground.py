import os


import jax
import numpy as np
from typing import TypeVar

Tensor = TypeVar("Tensor")

def comp(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
  return jax.lax.dot(jax.lax.exp(jax.lax.dot(a, b)), c)

size = 1000000000
jit = jax.jit(comp)
#key = jax.random.PRNGKey(0)
a = jax.numpy.zeros((size, 10))
b = jax.numpy.zeros((10, size))
c = jax.numpy.zeros((size,))

jit(a, b, c)
