from functools import partial
import click
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp

_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

if _gpu_dev is not None:
    print(f">>> GPU device information: {_gpu_dev}")
    print(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(_gpu_dev, True)


import sys
from pathlib import Path

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

# CUDA_VISIBLE_DEVICES="3" XLA_FLAGS="--xla_disable_hlo_passes=split-intermediate-tensors" python bench_knn.py 
# CUDA_VISIBLE_DEVICES="3" XLA_FLAGS="--xla_try_split_tensor_size=100MB" python bench_knn.py
# CUDA_VISIBLE_DEVICES="3" XLA_FLAGS="--xla_enable_hlo_passes_only=broadcast-simplifier,dot-order-optimizer" python bench_knn.py
# CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-knn" XLA_FLAGS="--xla_enable_hlo_passes_only=broadcast-simplifier,dot-order-optimizer --xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR}" python bench_knn.py


@click.command()
def main():
    seed = 111
    tf.random.set_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    n = 10000000
    m = 50
    d = 10
    k = 10
    dtype = np.float32
    data_points = np.array(rng.randn(n, d), dtype=dtype)
    query_points = np.array(rng.randn(m, d), dtype=dtype)

    results = run_jax_knn(data_points, query_points, k)
    print(results)
    print("finished")
    # run_tf_knn(data_points, query_points)


def run_tf_knn(data_points, query_points):
    knn_jit = tf.function(knn_tf, jit_compile=True)
    data_points = tf.convert_to_tensor(data_points)
    query_points = tf.convert_to_tensor(query_points)
    results = knn_jit(data_points, query_points)
    if isinstance(results, (list, tuple)):
        return [res.numpy() for res in results]
    else:
        return results.numpy()


def run_jax_knn(data_points, query_points, ktop: int):
    knn_jit = jax.jit(partial(knn_jax, k=ktop))
    # knn_jit = jax.jit(partial(knn, k=ktop))

    data_points = jnp.array(data_points)
    query_points = jnp.array(query_points)
    # ktop = jnp.array(ktop, int)
    results = knn_jit(data_points, query_points)
    if isinstance(results, (list, tuple)):
        return [res.block_until_ready() for res in results]
    else:
        return results.block_until_ready()


def knn_tf(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:

    """
    x2 = tf.reduce_sum(data_points ** 2, axis=-1)
    y2 = tf.reduce_sum(query_points ** 2, axis=-1)
    xy = tf.matmul(data_points, query_points, transpose_b=True)

    distances = x2[:, None] - 2.0 * xy + y2[None, :]
    distances_t = tf.transpose(distances)
    topk = tf.math.top_k(distances_t, k, sorted=False)
    return topk


def knn_jax(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:

    """
    x2 = jnp.sum(data_points ** 2, axis=-1)
    y2 = jnp.sum(query_points ** 2, axis=-1)
    xy = data_points @ query_points.T
    # square_distances = (data_points[:, None, :] - query_points[None, :, :]) ** 2
    # distances = jnp.sum(square_distances, axis=-1)

    distances = x2[:, None] - 2 * xy + y2[None, :]
    topk = jax.lax.top_k(distances.T, k)
    return topk


def knn(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:

    """
    # m = query_points.shape[0]
    # x2 = jnp.sum(data_points ** 2, axis=-1)
    # y2 = jnp.sum(query_points ** 2, axis=-1)
    # xy = data_points @ query_points.T
    square_distances = (data_points[:, None, :] - query_points[None, :, :]) ** 2
    distances = jnp.sum(square_distances, axis=-1)

    # distances = x2[:, None] - 2 * xy + y2[None, :]
    topk = jax.lax.top_k(distances.T, k)
    return topk


if __name__ == "__main__":
    main()
