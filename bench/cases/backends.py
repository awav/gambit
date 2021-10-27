import tensorflow as tf
import jax
from typing import Callable


def jit_compile(backend: str, func: Callable, xla: bool = True):
    if backend == "tf":
        fn_compiled = tf.function(func, jit_compile=xla)
        return fn_compiled
    elif backend == "jax":
        fn_compiled = jax.jit(func)
        return fn_compiled
    raise ValueError(f"Unknown backend '{backend}'. Choose either 'jax' or 'tf'")


def sync_cpu(backend: str, value):
    if backend == "jax":
        return value.block_until_ready()
    elif backend == "tf":
        return value.numpy()
    raise ValueError(f"Unknown backend '{backend}'. Choose either 'jax' or 'tf'")
