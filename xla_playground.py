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

if __name__ == "__main__":
    # main(False)
    # with_gradients()
    main_full_grad()
