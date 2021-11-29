import jax
import jax.numpy as jnp
import numpy as np


def layer(x, w, b):
    o = x @ w + b
    # return o
    # return jnp.exp(o)
    # return jnp.max(o)
    return jax.nn.relu(o)



def init(n: int = 10000, d: int = 100, h: int = 1000000, seed: int = 43):
    # prng = jax.random.PRNGKey(seed)
    # x = jax.random.uniform(prng, (n, d))
    # w = jax.random.uniform(prng, (d, h))
    # b = jax.random.uniform(prng, (h,))
    # return x, w, b
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    w = rng.rand(d, h)
    b = rng.rand(h)
    return x, w, b


def _init(n: int = 1000000, d: int = 10, seed: int = 43):
    # prng = jax.random.PRNGKey(seed)
    # x = jax.random.uniform(prng, (n, d))
    # w = jax.random.uniform(prng, (d, n))
    # b = jax.random.uniform(prng, (n,))
    prng = np.random.RandomState(seed)
    x = prng.rand(n, d)
    w = prng.rand(d, n)
    b = prng.rand(n).reshape(-1, 1)
    return x, w, b


def loss(x, w, b):
    output = layer(x, w, b)
    return jnp.mean(output)


def _loss(x, w, b):
    output = x @ w + b
    return jnp.mean(output)

x, w, b = init(h=100000)
loss_and_grad = jax.value_and_grad(loss, argnums=(1, 2))
loss_and_grad = jax.jit(loss_and_grad, inline=True)

value, grads = loss_and_grad(x, w, b)

print(f">>> Loss={value}, grads={grads}")
value.block_until_ready()

