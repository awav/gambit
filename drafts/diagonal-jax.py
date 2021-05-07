import jax
import jax.numpy as jnp
import jax.scipy as jsp


def genvector(n):
    return jnp.ones((n, 1))


def main(n: int):
    u = jnp.array(0.1)
    x = genvector(n)
    y = genvector(n)

    def run(x, y, u):
        z = x @ y.T
        v0 = 0.1 * jnp.concatenate([x, y], axis=1)
        v1 = u * jnp.eye(n)
        x = jsp.linalg.solve_triangular(z, v0, lower=True)
        x_times_v1 = v1 @ x
        return jnp.sum(x_times_v1)

    run_jit = jax.jit(run)
    res = [run_jit(x, y, u).block_until_ready() for _ in range(1)]
    return res


if __name__ == "__main__":
    main(1000)