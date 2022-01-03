import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

def forward(x):
  mlp = hk.nets.MLP([100000, 100, 10])
  return mlp(x)

forward = hk.transform(forward)

x = np.ones([50000, 28 * 28])

def _loss(x):
    rng = jax.random.PRNGKey(42)
    params = forward.init(rng, x)
    logits = forward.apply(params, rng, x)
    return logits


loss_fn = jax.jit(_loss, inline=True)
loss = loss_fn(x)

print("Finale")