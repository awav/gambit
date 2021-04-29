# VSCode's launch.json configuration for getting *.dot file of the graph before and after optimization
#
# {
#     "name": "Current File",
#     "type": "python",
#     "request": "launch",
#     "program": "${file}",
#     "console": "integratedTerminal",
#     "env": {
#         "TF_DUMP_GRAPH_PREFIX": "./xla-dump/",
#         "XLA_FLAGS": "--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/",
#         "TF_XLA_FLAGS": "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug",
#     },
#     "justMyCode": false,
# }
#
# Equivalent shell command
# TF_DUMP_GRAPH_PREFIX="./xla-dump/" XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-dump/" TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit --tf_xla_enable_xla_devices --tf_xla_clustering_debug" python xla_playground.py


from typing import TypeVar
import tensorflow as tf

Tensor = TypeVar("Tensor", bound=tf.Tensor)


@tf.function(experimental_compile=True)
def op(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    xy = x @ tf.transpose(y)
    xyz = xy @ z
    return xyz

def no_jit_op(x, y, z):
    xy = x @ tf.transpose(y)
    xyz = xy @ z
    return xyz

@tf.function(experimental_compile=True)
def more_complex_graph(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    a = x + y - z
    b = 3 * z
    c = tf.math.exp(y) - tf.math.log(tf.math.abs(x) + 0.001)
    ab = a @ tf.transpose(b)
    abc = ab @ c
    return abc / 5

# This doesn't work; the generated HLO for forward now explicitly requests to
# get the intermediate results, which breaks the optimization
def with_gradients():
    n, m = 1000, 2
    x = tf.random.normal((n, m))
    y = tf.random.normal((n, m))
    z = tf.random.normal((n, m))
    with tf.autodiff.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        g.watch(z)
        xyz = op(x, y, z)
    dx = g.gradient(xyz, x)
    # dy = g.gradient(xyz, y)
    # dz = g.gradient(xyz, z)
    return dx #, dy, dz

@tf.function(experimental_compile=True)
def gradients_full(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    with tf.autodiff.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        g.watch(z)

        xy = x @ tf.transpose(y)
        xyz = xy @ z

    dx = g.gradient(xyz, x)
    dy = g.gradient(xyz, y)
    dz = g.gradient(xyz, z)
    return (xyz, dx, dy, dz)

@tf.function(experimental_compile=True)
def loops(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    a = x
    for i in range(0, 9):
        if i % 2 == 0:
            a = a @ tf.transpose(y)
        else:
            a = a @ y
    return a @ z

@tf.function(experimental_compile=True)
def tf_loop(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    i = tf.constant(0)
    c = lambda i, _: tf.less(i, 10)
    b = lambda i, v: (tf.add(i, 1), v @ tf.transpose(y) @ y)
    r = tf.while_loop(c, b, [i, x])
    return r

def main(simple=True):
    n, m = 1000, 2
    x = tf.random.normal((n, m))
    y = tf.random.normal((n, m))
    z = tf.random.normal((n, m))
    if simple:
        res = op(x, y, z)
        print(f"Op result = {res.numpy()}")
        print(f"Delta = {(res - no_jit_op(x, y, z)).numpy()}")
    else:
        res = more_complex_graph(x, y, z)
        print(f"Op result = {res.numpy()}")

def main_full_grad():
    n, m = 1000, 2
    x = tf.random.normal((n, m))
    y = tf.random.normal((n, m))
    z = tf.random.normal((n, m))
    xyz, dx, dy, dz = gradients_full(x, y, z)

def main_while_loop():
    n, m = 1000, 2
    x = tf.random.normal((n, m))
    y = tf.random.normal((n, m))
    z = tf.random.normal((n, m))
    # loops(x, y, z)
    tf_loop(x, y, z)

# $exp(||X-Y||^2) v$ (where exp is pointwise)
@tf.function(experimental_compile=True)
def test_dist_matrix(x: Tensor, y: Tensor, v: Tensor) -> Tensor:
    xx = tf.reduce_sum(tf.square(x), 1)
    yy = tf.reduce_sum(tf.square(y), 1)

    xx = tf.reshape(xx, (-1, 1))
    yy = tf.reshape(yy, (1, -1))

    D = tf.sqrt(tf.maximum(xx + yy - 2.0 * tf.matmul(x, y, False, True), 0.0))

    e = tf.math.exp(D)
    return e @ v

def test_dist_matrix_no_xla(x: Tensor, y: Tensor, v: Tensor) -> Tensor:
    xx = tf.reduce_sum(tf.square(x), 1)
    yy = tf.reduce_sum(tf.square(y), 1)

    xx = tf.reshape(xx, (-1, 1))
    yy = tf.reshape(yy, (1, -1))

    D = tf.sqrt(tf.maximum(xx + yy - 2.0 * tf.matmul(x, y, False, True), 0.0))

    e = tf.math.exp(D)
    return e @ v

def main_dist_matrix():
    n, m, l = 2000, 3, 2
    x = tf.random.normal((n, m))
    y = tf.random.normal((n, m))
    v = tf.random.normal((n, l))
    res = test_dist_matrix(x, y, v).numpy()
    res_no_xla = test_dist_matrix_no_xla(x, y, v).numpy()
    print(res.shape)
    print(res - res_no_xla)

@tf.function(experimental_compile=True)
def most_simple_example(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    outer = x @ tf.transpose(y)
    outer_mapped = tf.math.sin(outer)
    return outer_mapped @ z

def most_simple_example_no_xla(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    outer = x @ tf.transpose(y)
    outer_mapped = tf.math.sin(outer)
    return outer_mapped @ z

@tf.function(experimental_compile=True)
def most_simple_with_grad(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    with tf.autodiff.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        g.watch(z)
        outer = x @ tf.transpose(y)
        outer_mapped = tf.math.sin(outer)
        final = outer_mapped @ z
    dx = g.gradient(final, x)
    dy = g.gradient(final, y)
    dz = g.gradient(final, z)
    return (final, dx, dy, dz)

def most_simple_with_grad_no_xla(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    with tf.autodiff.GradientTape(persistent=True) as g:
        g.watch(x)
        g.watch(y)
        g.watch(z)
        outer = x @ tf.transpose(y)
        outer_mapped = tf.math.sin(outer)
        final = outer_mapped @ z
    dx = g.gradient(final, x)
    dy = g.gradient(final, y)
    dz = g.gradient(final, z)
    return (final, dx, dy, dz)

def main_simple_example():
    x = tf.random.normal((2000, 2))
    y = tf.random.normal((10000, 2))
    z = tf.random.normal((10000, 2))
    # res11 = most_simple_with_grad(x, y, z)
    # res21 = most_simple_example_no_xla(x, y, z)
    # print(res11[0].numpy() - res21[0].numpy())
    res12 = most_simple_example(x, y, z)
    res22 = most_simple_example_no_xla(x, y, z)
    print(res12.numpy() - res22.numpy())
    # print(res21.numpy() - res22.numpy())

@tf.function(experimental_compile=True)
def nice_dist_matrix(x: Tensor, y: Tensor) -> Tensor:
    diff = x[None, :, :] - y[:, None, :]
    dist = diff ** 2
    return tf.reduce_sum(dist, axis=2)

def test_nice_dist_matrix():
    x = tf.random.normal((4, 2))
    y = tf.random.normal((4, 2))
    print(nice_dist_matrix(x, y).numpy())

if __name__ == "__main__":
    # main(False)
    # with_gradients()
    # main_full_grad()
    main_dist_matrix()
    # main_simple_example()
    # test_nice_dist_matrix()
