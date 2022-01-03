import numpy as np
import jax
import jax.numpy as jnp
import optax
from jax import grad, jit, vmap, value_and_grad
from jax import random
from jax.experimental import optimizers
from tensorflow.python.ops.gen_batch_ops import batch


def init_mlp(sizes, seed):
    """
    Initialize the weights of all layers of a linear layer network
    """
    rng = np.random.RandomState(seed)

    def init_layer(m, n, scale=1e-2):
        w_init = scale * rng.randn(n, m)
        b_init = scale * rng.randn(n)
        return w_init, b_init

    init_conf = zip(sizes[:-1], sizes[1:])
    return [init_layer(m, n) for m, n in init_conf]


def relu_layer(w, b, x):
    return jax.nn.relu(w @ x + b)


def forward_pass(params, in_array):
    """ Compute the forward pass for each example individually """
    act = in_array

    for w, b in params[:-1]:
        act = relu_layer(w, b, act)

    final_w, final_b = params[-1]
    logits = final_w @ act + final_b
    return logits - jax.nn.logsumexp(logits)


def random_data():
    n: int = 1000000
    d: int = 28 * 28
    c: int = 10
    seed: int = 10
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d)
    y = rng.randint(0, c, n)
    y_one_hot = y[:, None] == np.arange(c)
    return x, y_one_hot


def mnist_data():
    import tensorflow_datasets as tfds
    import tensorflow as tf

    devs = tf.config.get_visible_devices()
    for dev in devs:
        if dev.device_type != "GPU":
            continue
        tf.config.experimental.set_memory_growth(dev, True)

    c = 10
    ds = tfds.load("mnist", split="train[:80%]", batch_size=-1)
    ds_test = tfds.load("mnist", split="train[80%:]", batch_size=-1)
    ds = tfds.as_numpy(ds)
    ds_test = tfds.as_numpy(ds_test)

    def datanorm(ds):
        x, y = ds["image"], ds["label"]
        n = x.shape[0]
        x = np.array(x.reshape(n, -1) / 255.0, np.float32)
        y = np.array(y[:, None] == np.arange(c), np.float32)
        return x, y
    
    train = datanorm(ds)
    test = datanorm(ds_test)

    del ds, ds_test
    del tf, tfds
    return train, test


def main():
    seed = 1
    (x, y), (x_test, y_test) = mnist_data()

    cut = 1000
    x = x[:cut]
    y = y[:cut]
    x_test = x_test[:cut]
    y_test = y_test[:cut]

    d = x.shape[-1]
    # layer_sizes = [d, 1000000, 512, 10]
    layer_sizes = [d, 1000000, 10]

    step_size = 1e-3
    num_epochs = 100

    params = init_mlp(layer_sizes, seed)
    batch_forward = vmap(forward_pass, in_axes=(None, 0), out_axes=0)
    lr_schedule = optimizers.polynomial_decay(0.01, num_epochs, 0.0005)
    opt_init, opt_update, get_params = optimizers.adam(lr_schedule)
    opt_state = opt_init(params)

    def _accuracy(params, x, y):
        target_class = jnp.argmax(y, axis=-1)
        predicted_class = jnp.argmax(batch_forward(params, x), axis=-1)
        acc = jnp.mean(predicted_class == target_class)
        return acc

    def _loss(params, in_arrays, targets):
        """ Compute the multi-class cross-entropy loss """
        logits = batch_forward(params, in_arrays)
        entropy = optax.softmax_cross_entropy(logits, targets)
        return jnp.sum(entropy)
    
    def _update_with_grad(params, x, y, opt_state):
        """ Compute the gradient for a batch and update the parameters """
        value, grads = value_and_grad(_loss, argnums=0)(params, x, y)
        opt_state = opt_update(0, grads, opt_state)
        return get_params(opt_state), opt_state, value
    
    update_with_grads = jax.jit(_update_with_grad, inline=True)
    accuracy = jax.jit(_accuracy, inline=True)

    for epoch in range(num_epochs):
        params, opt_state, loss = update_with_grads(params, x, y, opt_state)
        test_acc = accuracy(params, x_test, y_test)
        print(f"{epoch}! loss={np.array(loss)}, test_acc={np.array(test_acc)}")

    print("Finished!")


if __name__ == "__main__":
    main()
