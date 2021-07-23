from typing import Union
import tensorflow as tf
import gpflow

Tensor = tf.Tensor


def create_kernel(name: str, dim: int, dtype=None):
    with gpflow.config.as_context(gpflow.config.Config(float=dtype)):
        if name == "se":
            lengthscale = [0.1] * dim
            return gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=lengthscale)
        if name == "matern32":
            lengthscale = [0.1] * dim
            return gpflow.kernels.Matern32(variance=0.1, lengthscales=lengthscale)
        elif name == "linear":
            return gpflow.kernels.Linear(variance=0.1)
        raise ValueError(f"Unknown kernel {name}")


def kernel_vector_product(
    kernel, a_input: Tensor, b_input: Union[Tensor, None], v_vector: Tensor
) -> Tensor:
    k_ab = kernel.K(a_input, b_input)
    small_tensor = k_ab @ v_vector
    return small_tensor
