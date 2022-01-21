from multiprocessing.sharedctypes import Value
from typing import Callable, Union, Dict
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp

import torch
import pykeops
from pykeops.torch import LazyTensor

import sys
from pathlib import Path

_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

if _gpu_dev is not None:
    click.echo(f">>> GPU device information: {_gpu_dev}")
    click.echo(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(_gpu_dev, True)

curdir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(curdir)

from clitypes import LogdirPath
from bench_utils import BenchRunner, read_h5_into_dict, store_dict_as_h5, parse_name

Backend = Literal["jax", "keops"]
Distance = Literal["L2", "L1", "cosine"]
PathLike = Union[Path, str]


__ann_dataset_path = Path("~/code/ann-benchmarks/data").expanduser().resolve()
__default_logdir = Path(curdir, "logs_knn_default")
__download_dir = "~/.datasets"

backend_choices = click.Choice(["jax", "keops", "tf", "jax-pure", "tf-pure"])
distance_choices = click.Choice(["L2", "L1", "cosine"])


class DatasetType(click.ParamType):
    name = "dataset_type"

    def convert(self, value, param, ctx):
        datasets = {"random", "mnist", "fashion", "glove50", "glove100", "glove200"}
        names = {"n": "dataset_size", "m": "query_size", "d": "dim_size"}
        try:
            parsed_dict = parse_name(value)
            name = parsed_dict["name"]
            if name not in datasets:
                raise RuntimeError(f"Not supported dataset {name}")
            parsed_map_dict = {
                names[key]: int(value) for key, value in parsed_dict.items() if key in names
            }
            return {"dataset_name": name, "dataset_config": value, **parsed_map_dict}
        except Exception as ex:
            self.fail(f"{value} is not a valid dataset type", param, ctx)

    def __repr__(self):
        return "DatasetType"


@click.command()
@click.option("-s", "--seed", type=int, default=None)
@click.option("-r", "--repeat", type=int, default=1, help="Number of experiment repeatitions")
@click.option("-w", "--warmup", type=int, default=1, help="Number of warm-up iterations")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_logdir)
@click.option("-c", "--distance", type=distance_choices, default="L2")
@click.option("-k", "--topk", type=int, default=10)
@click.option("-b", "--backend", type=backend_choices, default="jax")
@click.option("-d", "--dataset", type=DatasetType())
def main(
    backend: Backend,
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    distance: Distance,
    dataset: Dict,
    topk: int,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    backend_full = backend
    if backend_full.endswith("-pure"):
        backend = backend_full.split("-")[0]

    prepare_backend(backend, logdir)
    upd_dataset_info, exp_args_fn = create_exp_args(backend, dataset, seed)
    exp_func = create_exp_knn(backend, topk, distance)

    bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
    results = bench_runner.bench(exp_func, exp_args_fn)

    bench_table = dict(
        backend=backend_full,
        seed=seed,
        warmup=warmup,
        repeat=repeat,
        distance=distance,
        topk=topk,
        **upd_dataset_info,
        **results,
    )

    filepath = Path(logdir, "bench.h5")
    store_dict_as_h5(bench_table, filepath)

    (elap_mu, elap_std) = results["elapsed_stats"]
    (mem_mu, mem_std) = results["mem_stats"]

    click.echo(
        "[Bench] Total stat, "
        f"spent_avg={elap_mu}, spent_std={elap_std}, "
        f"mem_avg={mem_mu}, mem_std={mem_std}"
    )


def prepare_backend(backend: Backend, logdir: PathLike):
    if backend == "jax" or backend == "tf":
        return
    elif backend == "keops":
        pykeops.set_bin_folder(str(Path(logdir, "keops")))
        pykeops.clean_pykeops()
        return
    raise ValueError(f"Unknown backend passed: {backend}")


def create_exp_knn(backend: Backend, k: int, distance: Distance) -> Callable:
    if backend == "jax":
        return prepare_jax_knn(k, distance)
    elif backend == "keops":
        return prepare_keops_knn(k, distance)
    elif backend == "tf":
        return prepare_tf_knn(k, distance)
    raise NotImplementedError(f"Uknown backend passed: {backend}")


def create_exp_args(backend: Backend, dataset: Dict, seed: int):
    output_dataset = dataset.copy()
    name = dataset["dataset_name"]
    orig_name = dataset["dataset_config"]

    if name == "random":
        must_have_keys = {"dataset_size", "query_size", "dim_size"}
        if not all([key in dataset for key in must_have_keys]):
            raise ValueError(f"Some keys are missing in {orig_name}")
        dtype = np.float32
        rng = np.random.RandomState(seed)
        n = dataset["dataset_size"]
        m = dataset["query_size"]
        d = dataset["dim_size"]

        def random_args_fn():
            rnd_data_points = np.array(rng.randn(n, d), dtype=dtype)
            rnd_query_points = np.array(rng.randn(m, d), dtype=dtype)
            return rnd_data_points, rnd_query_points

        args_fn = random_args_fn

    elif name == "mnist" or name == "fashion":
        import torchvision

        dataset_classes = {
            "mnist": torchvision.datasets.MNIST,
            "fashion": torchvision.datasets.FashionMNIST,
        }
        if "query_size" not in dataset:
            raise ValueError(f"Query size is missing in {orig_name}")

        m = dataset["query_size"]
        dataset_class = dataset_classes[name]
        ds = dataset_class(__download_dir, download=True)
        data_points = np.array(ds.data.cpu().numpy(), dtype=np.float32)
        data_points /= 255.0
        n = data_points.shape[0]
        data_points = data_points.reshape(n, -1)
        d = data_points.shape[-1]
        query_indices = np.random.choice(data_points.shape[0], size=m)
        query_points = data_points[query_indices, ...]

        output_dataset["dataset_size"] = data_points.shape[0]
        output_dataset["dim_size"] = data_points.shape[-1]

        def mnist_args_fn():
            return data_points, query_points

        args_fn = mnist_args_fn
    elif name.startswith("glove"):
        glove_filepath = Path(__ann_dataset_path, f"{name}-angular.hdf5")
        dataset_dict = read_h5_into_dict(glove_filepath)
        data_points = dataset_dict["train"]
        query_points = dataset_dict["test"]

        output_dataset["query_size"] = data_points.shape[0]
        output_dataset["dataset_size"] = data_points.shape[0]
        output_dataset["dim_size"] = data_points.shape[-1]

        def glove_args_fn():
            return data_points, query_points

        args_fn = glove_args_fn
    else:
        raise NotImplementedError(f"Uknown dataset passed: {dataset}")

    if backend == "jax":
        return output_dataset, prepare_jax_args_fn(args_fn)
    elif backend == "keops":
        return output_dataset, prepare_keops_args_fn(args_fn)
    elif backend == "tf":
        return output_dataset, prepare_tf_args_fn(args_fn)

    raise NotImplementedError(f"Uknown backend passed: {backend}")


# TF implementation


def prepare_tf_knn(k: int, distance: Distance):
    knn_funcs = {
        "L2": knn_l2_tf,
        "L1": knn_l1_tf,
        "cosine": knn_cosine_tf,
    }
    chosen_knn = knn_funcs[distance]

    def knn_func(*args, **kwargs):
        return chosen_knn(*args, k=k, **kwargs)

    knn_func_jit = tf.function(knn_func, jit_compile=True)

    def _func(*args, **kwargs):
        result = knn_func_jit(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            return [r.numpy() for r in result]
        return result.numpy()

    return _func


def prepare_tf_args_fn(args_fn):
    ctt = tf.convert_to_tensor

    def jax_args_fn():
        args = args_fn()
        return [ctt(arg) for arg in args]

    return jax_args_fn


def knn_l2_tf(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:
        L2 distance matrix
    """
    x2 = tf.reduce_sum(data_points ** 2, axis=-1)
    y2 = tf.reduce_sum(query_points ** 2, axis=-1)
    xy = tf.matmul(data_points, query_points, transpose_b=True)

    distances = x2[:, None] - 2.0 * xy + y2[None, :]
    distances_t = tf.transpose(distances)
    _, topk_indices = tf.math.top_k(-distances_t, k)
    return topk_indices


def knn_l1_tf(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:
        L1 distance matrix
    """
    abs_distances = tf.math.abs(data_points[:, None, :] - query_points[None, :, :])
    distances = tf.reduce_sum(abs_distances, axis=-1)
    distances_t = tf.transpose(distances)
    _, topk_indices = tf.math.top_k(-distances_t, k)
    return topk_indices


def knn_cosine_tf(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:
        Angular distance matrix
    """
    xy_dot = tf.matmul(data_points, query_points, transpose_b=True)
    x_norm = tf.reduce_sum(data_points ** 2, axis=-1)
    y_norm = tf.reduce_sum(query_points ** 2, axis=-1)
    norms = tf.math.sqrt(x_norm[:, None] * y_norm)
    distances = 1.0 - xy_dot / norms
    distances_t = tf.transpose(distances)
    _, topk_indices = tf.math.top_k(-distances_t, k)
    return topk_indices


# JAX implementation


def prepare_jax_knn(k: int, distance: Distance):
    knn_funcs = {
        "L2": knn_l2_jax,
        "L1": knn_l1_jax,
        "cosine": knn_cosine_jax,
    }
    chosen_knn_jax = knn_funcs[distance]

    def knn_func(*args, **kwargs):
        return chosen_knn_jax(*args, k=k, **kwargs)

    knn_func_jit = jax.jit(knn_func)

    def _func(*args, **kwargs):
        result = knn_func_jit(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            return [r.block_until_ready() for r in result]
        return result.block_until_ready()

    return _func


def prepare_jax_args_fn(args_fn):
    def jax_args_fn():
        args = args_fn()
        return [jnp.array(arg) for arg in args]

    return jax_args_fn


def knn_l2_jax(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:
        L2 distance matrix
    """
    x2 = jnp.sum(data_points ** 2, axis=-1)
    y2 = jnp.sum(query_points ** 2, axis=-1)
    xy = data_points @ query_points.T
    # square_distances = (data_points[:, None, :] - query_points[None, :, :]) ** 2
    # distances = jnp.sum(square_distances, axis=-1)

    distances = x2[:, None] - 2 * xy + y2[None, :]
    _, topk_indices = jax.lax.top_k(-distances.T, k)
    return topk_indices


def knn_l1_jax(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:
        L1 distance matrix
    """
    abs_distances = jnp.abs(data_points[:, None, :] - query_points[None, :, :])
    distances = jnp.sum(abs_distances, axis=-1)
    _, topk_indices = jax.lax.top_k(-distances.T, k)
    return topk_indices


def knn_cosine_jax(data_points, query_points, k: int = 10):
    """
    Args:
        data_points: [N, D] tensor
        query_points: [M, D] tensor
    Return:
        Angular distance matrix
    """
    xy_dot = data_points @ query_points.T
    x_norm = jnp.sum(data_points ** 2, axis=-1)
    y_norm = jnp.sum(query_points ** 2, axis=-1)
    norms = jnp.sqrt(x_norm[:, None] * y_norm)
    distances = 1.0 - xy_dot / norms
    _, topk_indices = jax.lax.top_k(-distances.T, k)
    return topk_indices


# KeOps implementation


def prepare_keops_args_fn(args_fn):
    def keops_args_fn():
        args = args_fn()
        if _gpu_dev is not None:
            return [torch.tensor(arg).cuda() for arg in args]
        return [torch.tensor(arg) for arg in args]

    return keops_args_fn


def prepare_keops_knn(k: int, distance: Distance):
    knn_funcs = {
        "L2": knn_l2_keops,
        "L1": knn_l1_keops,
        "cosine": knn_cosine_keops,
    }
    chosen_knn_jax = knn_funcs[distance]

    def knn_func(*args, **kwargs):
        return chosen_knn_jax(*args, k=k, **kwargs)

    def _func(*args, **kwargs):
        result = knn_func(*args, **kwargs)
        if isinstance(result, (list, tuple)):
            return [r.cpu().numpy() for r in result]
        return result.cpu().numpy()

    return _func


def _lazy_input_tensors(data_points, query_points):
    n = data_points.shape[0]
    d = data_points.shape[-1]
    m = query_points.shape[0]
    data_points_lazy = LazyTensor(data_points.view(n, 1, d))
    query_points_lazy = LazyTensor(query_points.view(1, m, d))
    return data_points_lazy, query_points_lazy


def knn_l1_keops(data_points, query_points, k: int = 10):
    data_points_lazy, query_points_lazy = _lazy_input_tensors(data_points, query_points)
    distance = (data_points_lazy - query_points_lazy).abs().sum(dim=2)  # diff from L2
    indices = distance.argKmin(K=k, dim=1)
    return indices


def knn_l2_keops(data_points, query_points, k: int = 10):
    data_points_lazy, query_points_lazy = _lazy_input_tensors(data_points, query_points)
    distance = ((data_points_lazy - query_points_lazy) ** 2).sum(dim=2)  # diff from L1
    indices = distance.argKmin(K=k, dim=1)
    return indices


def knn_cosine_keops(data_points, query_points, k: int = 10):
    data_points_lazy, query_points_lazy = _lazy_input_tensors(data_points, query_points)
    distances = 1 - (data_points_lazy | query_points_lazy)
    indices = distances.argKmin(K=k, dim=1)
    return indices


if __name__ == "__main__":
    main()
