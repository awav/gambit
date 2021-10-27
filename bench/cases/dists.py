from typing import Callable, Optional, Union
import tensorflow as tf
import jax
from jax import numpy as jnp
from jax import random as jrnd

Tensor = Union[jnp.ndarray, tf.Tensor]


def squared_eucledian_distance(x: jnp.ndarray, y: jnp.ndarray):
    diff = x[:, None, ...] - y[None, ...]
    return jnp.sum(diff ** 2, axis=-1)


def se_kernel_product(
    x: jnp.ndarray,
    y: jnp.ndarray,
    v: jnp.ndarray,
    lengthscale: jnp.ndarray,
    variance: jnp.ndarray,
) -> jnp.ndarray:
    """
    Args:
        x: Matrix with expected [N, D] shape
        y: Matrix with expected [M, D] shape
        v: Vector of M elements
    """
    dist_matrix = squared_eucledian_distance(x, y)
    kernel = variance * jnp.exp(dist_matrix)
    return kernel @ v


def create_dist_function_evaluation(
    dim: int,
    a_size: int,
    b_size: int,
    seed: int = 0,
    dtype: Optional[jnp.dtype] = None,
) -> Callable:
    key = jax.random.PRNGKey(seed)
    a = jax.random.uniform(key, (a_size, dim), dtype=dtype)
    b = jax.random.uniform(key, (b_size, dim), dtype=dtype)
    v = jax.random.uniform(key, (b_size,), dtype=dtype)

    lengthscale = jnp.ones((dim,), dtype=dtype) * 0.1
    variance = jnp.array(0.1, dtype=dtype)

    jitted_func = jax.jit(se_kernel_product)

    def eval_func():
        return jitted_func(a, b, v, lengthscale, variance)

    return eval_func
