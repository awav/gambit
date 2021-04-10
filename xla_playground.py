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

def main():
    n, m = 1000, 2
    x = tf.random.normal((n, m))
    y = tf.random.normal((n, m))
    z = tf.random.normal((n, m))
    res = op(x, y, z)
    print(f"Op result = {res.numpy()}")
    print(f"Delta = {(res - no_jit_op(x, y, z)).numpy()}")


if __name__ == "__main__":
    main()