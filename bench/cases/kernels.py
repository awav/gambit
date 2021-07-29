from typing import Union, List
import tensorflow as tf
import gpflow

Tensor = tf.Tensor


def create_kernel(name: str, dim: int, dtype=None):
    dtype = tf.float64 if dtype is None else dtype
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


def kernel_vector_product_with_grads(
    kernel, a_input: Tensor, b_input: Union[Tensor, None], v_vector: Tensor
) -> List[Tensor]:
    variables = list(kernel.trainable_variables)
    with tf.GradientTape() as tape:
        k_ab = kernel.K(a_input, b_input)
        small_tensor = k_ab @ v_vector
        loss = tf.reduce_sum(small_tensor)
    grads = tape.gradient(loss, variables)
    result = [loss, *grads]
    return result


def kernel_vector_product(
    kernel, a_input: Tensor, b_input: Union[Tensor, None], v_vector: Tensor
) -> List[Tensor]:
    k_ab = kernel.K(a_input, b_input)
    result = k_ab @ v_vector
    return result