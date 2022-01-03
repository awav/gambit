import jax
import jax.numpy as jnp
import numpy as np


def layer(x, w, b=None):
    xw = x @ w 
    if b is None:
        return xw
    o = xw + b
    return jax.nn.relu(o)



def init(n: int = 10000, d: int = 100, h1: int = 1000000, h2: int = 10, seed: int = 43):
    # prng = jax.random.PRNGKey(seed)
    # x = jax.random.uniform(prng, (n, d))
    # w = jax.random.uniform(prng, (d, h))
    # b = jax.random.uniform(prng, (h,))
    # return x, w, b
    rng = np.random.RandomState(seed)
    x = rng.rand(n, d)
    w1 = rng.rand(d, h1)
    w2 = rng.rand(h1, h2)
    b = rng.rand(h1)
    return x, w1, w2, b


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


def loss(x, w1, w2, b):
    layer_1_output = layer(x, w1, b)
    output = layer(layer_1_output, w2)
    return jnp.mean(output)


def _loss(x, w, b):
    output = x @ w + b
    return jnp.mean(output)


x, w1, w2, b = init(h1=100000)
loss_and_grad = jax.value_and_grad(loss, argnums=(1, 2, 3))
loss_and_grad = jax.jit(loss_and_grad, inline=True)

value, grads = loss_and_grad(x, w1, w2, b)

print(f">>> Loss={value}, grads={grads}")
value.block_until_ready()

