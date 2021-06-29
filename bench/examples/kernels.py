import tensorflow as tf
import gpflow

Tensor = tf.Tensor


def create_kernel(name: str, dim: int):
    if name == "se":
        lengthscale = [0.1] * dim
        return gpflow.kernels.SquaredExponential(variance=0.1, lengthscales=lengthscale)
    elif name == "linear":
        return gpflow.kernels.Linear(variance=0.1)
    raise ValueError(f"Unknown kernel {name}")


def kernel_vector_product(
    kernel, a_input: Tensor, b_input: Tensor, v_vector: Tensor
) -> Tensor:
    k_ab = kernel(a_input, b_input)
    small_tensor = k_ab @ v_vector
    return small_tensor
