from collections import namedtuple
import tensorflow as tf
import numpy as np

T = tf.Tensor


def add_diagonal(matrix: T, diagonal_scalar: T) -> T:
    n = matrix.shape[0]
    dtype = matrix.dtype
    return matrix + diagonal_scalar * tf.eye(n, dtype=dtype)


def squared_exponential(vec_x: T, vec_y: T, lengthscale: T) -> T:
    diff = (vec_x[:, None, :] - vec_y[None, :, :]) ** 2
    distance = tf.reduce_sum(diff / lengthscale[None, None, :] ** 2, axis=-1)
    return tf.exp(distance)


def conjugate_gradient(
    matrix: T, rhs: T, initial_solution: T, epsilon: T, max_iter: int, repeat: int
) -> T:
    """
    Find a solution to `Kx = b`, where `K \in R^{N \times N}`, `x \in R^{N}`, and `b \in R^{N}`.

    Args:
        matrix: Matrix with shape [N, N]. K matrix in `Kx = b`
        rhs: Vector of length [N, B]. b vector in `Kx = b`
        initial_solution: Vector of length [N]. `x_0` initial guess vector in `K x_0 = b`
    
    Returns:
        Solution vector `x` to `Kx = b` expression.
    """
    K = matrix
    b = rhs
    x0 = initial_solution

    r = b - K @ x0
    d = r
    delta0 = tf.reduce_sum(r ** 2)

    State = namedtuple("State", "i, x, r, d, delta")

    def cond(state):
        return state.i < max_iter and state.delta >= delta0 * (epsilon ** 2)

    def body(state):
        q = K @ state.d
        alpha = state.delta / tf.reduce_sum(state.d * q)
        x_next = state.x + alpha * state.d

        if state.i != 0 and state.i % repeat == 0:
            r_next = b - K @ x_next
        else:
            r_next = state.r - alpha * q

        delta_next = tf.matmul(r_next, r_next, transpose_a=True)[0, 0]
        beta = delta_next / state.delta
        d_next = r_next + beta * state.d
        i_next = state.i + 1
        return [State(i=i_next, x=x_next, r=r_next, d=d_next, delta=delta_next)]

    state_0 = State(i=0, x=x0, r=r, d=d, delta=delta0)
    states = tf.while_loop(cond, body, [state_0])
    final_state = states[0]
    return final_state.x


def cg_example(
    x: T,
    y: T,
    b: T,
    l: T,
    s: T,
    e: T,
    max_iters: int,
    reset_iter: int,
) -> T:
    """
    x: Vector with shape [N, D]
    y: Vector with shape [N, D]
    b: Vector of RHS in `Kx = b` expression.
    l: Vector of K's hyperparameters - lengthscales.
    s: Hyperparameter scalar \sigma^2.
    e: Conjugate gradient error scalar.
    max_iters: Maximum number of CG iterations.
    reset_iter: Reset iteration for CG.
    """
    dtype = x.dtype
    with tf.name_scope("kernel"):
        K = squared_exponential(x, y, l)
    with tf.name_scope("add_diagonal"):
        K_sigmaI = add_diagonal(K, s)
    x0 = tf.zeros_like(b, dtype=dtype)
    with tf.name_scope("conjugate_gradient"):
        return conjugate_gradient(K_sigmaI, b, x0, e, max_iters, reset_iter)


def main():
    tf.random.set_seed(111)
    cg_example_compiled = tf.function(cg_example, experimental_compile=True)
    # cg_example_compiled = cg_example

    dtype = tf.float64
    d = 2
    n = 100
    x = tf.random.uniform((n, d), dtype=dtype)
    b = tf.ones([n, 1], dtype=dtype)
    lengthscale = 3.0 * tf.ones([d], dtype=dtype)
    sigma_2 = 5.0
    error = 0.1
    max_iters = n // 2
    repeat_iter = n // 4

    res = cg_example_compiled(
        x,
        x,
        b,
        lengthscale,
        sigma_2,
        error,
        max_iters,
        repeat_iter,
    )

    res_numpy = res.numpy()
    print("the end")


if __name__ == "__main__":
    main()
