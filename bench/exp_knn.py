from typing import Callable, Union
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
    print(f">>> GPU device information: {_gpu_dev}")
    print(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(_gpu_dev, True)

curdir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(curdir)

from clitypes import LogdirPath
from bench_utils import BenchRunner, read_h5_into_dict, store_dict_as_h5, parse_name

Backend = Literal["jax", "keops"]
Distance = Literal["L2", "L1", "cosine"]
PathLike = Union[Path, str]


__default_logdir = Path(curdir, "logs_knn_default")
__download_dir = "~/.datasets"

backend_choices = click.Choice(["jax", "keops"])
distance_choices = click.Choice(["L2", "L1", "cosine"])


@click.command()
@click.option("-s", "--seed", type=int, default=None)
@click.option("-r", "--repeat", type=int, default=1, help="Number of experiment repeatitions")
@click.option("-w", "--warmup", type=int, default=1, help="Number of warm-up iterations")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_logdir)
@click.option("-c", "--distance", type=distance_choices, default="L2")
@click.option("-k", "--topk", type=int, default=10000)
@click.option("-b", "--backend", type=backend_choices, default="jax")
@click.option("-d", "--dataset", type=str)
def main(
    backend: Backend,
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    distance: Distance,
    dataset: str,
    topk: int,
):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    prepare_backend(backend, logdir)
    dataset_name, args_info, exp_args_fn = create_exp_args(backend, dataset, seed)
    exp_func = create_exp_knn(backend, topk, distance)

    bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
    results = bench_runner.bench(exp_func, exp_args_fn)

    bench_table = dict(
        backend=backend,
        seed=seed,
        warmup=warmup,
        repeat=repeat,
        distance=distance,
        dataset=dataset,
        dataset_name=dataset_name,
        topk=topk,
        **args_info,
        **results,
    )

    filepath = Path(logdir, "bench.h5")
    store_dict_as_h5(bench_table, filepath)

    (elap_mu, elap_std) = results["elapsed_stats"]
    (mem_mu, mem_std) = results["mem_stats"]

    print(
        "[Bench] Total stat, "
        f"spent_avg={elap_mu}, spent_std={elap_std}, "
        f"mem_avg={mem_mu}, mem_std={mem_std}"
    )


def prepare_backend(backend: Backend, logdir: PathLike):
    if backend == "jax":
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
    raise NotImplementedError(f"Uknown backend passed: {backend}")


def create_exp_args(backend: Backend, dataset: str, seed: int):
    if dataset.startswith("random"):
        conf = parse_name(dataset)
        dtype = np.float32
        n = int(conf["n"])
        m = int(conf["m"])
        d = int(conf["d"])
        conf = {"dataset_size": n, "query_size": m, "dim_size": d}
        dataset_name: Literal["random"] = "random"
        rng = np.random.RandomState(seed)

        def random_args_fn():
            data_points = np.array(rng.randn(n, d), dtype=dtype)
            query_points = np.array(rng.randn(m, d), dtype=dtype)
            return data_points, query_points

        args_fn = random_args_fn

    elif dataset.startswith("mnist") or dataset.startswith("fashion"):
        import torchvision

        conf = parse_name(dataset)
        dataset_name: Literal["mnist", "fashion"] = conf["name"]

        dataset_classes = {
            "mnist": torchvision.datasets.MNIST,
            "fashion": torchvision.datasets.FashionMNIST,
        }
        if dataset_name not in dataset_classes:
            raise RuntimeError(f"Unknown parsed dataset name: {dataset_name}")

        dataset_class = dataset_classes[dataset_name]
        ds = dataset_class(__download_dir, download=True)
        data_points = np.array(ds.data.cpu().numpy(), dtype=np.float32)
        data_points /= 255.0
        n = data_points.shape[0]
        data_points = data_points.reshape(n, -1)
        d = data_points.shape[-1]
        m = int(conf["m"])
        query_indices = np.random.choice(data_points.shape[0], size=m)
        query_points = data_points[query_indices, ...]

        def mnist_args_fn():
            return data_points, query_points

        args_fn = mnist_args_fn
    elif dataset.startswith("glove"):
        datasets = {"glove-50", "glove-100", "glove-200"}
        if dataset not in datasets:
            raise RuntimeError(f"Unknown dataset name: {dataset}")
        dataset_name = dataset
        # TODO(awav): change hard-coded path to the directory
        glove_dirpath = Path("~/code/ann-benchmarks/data").expanduser().resolve()
        glove_filepath = Path(glove_dirpath, f"{dataset_name}-angular.hdf5")  
        dataset_dict = read_h5_into_dict(glove_filepath)
        data_points = dataset_dict["train"]
        query_points = dataset_dict["test"]

        def glove_args_fn():
            return data_points, query_points
        
        args_fn = glove_args_fn
    else:
        raise NotImplementedError(f"Uknown dataset passed: {dataset}")

    n: int = data_points.shape[0]
    d: int = data_points.shape[-1]
    m: int = query_points.shape[0]
    conf = {"dataset_size": n, "query_size": m, "dim_size": d}

    if backend == "jax":
        return dataset_name, conf, prepare_jax_args_fn(args_fn)
    elif backend == "keops":
        return dataset_name, conf, prepare_keops_args_fn(args_fn)

    raise NotImplementedError(f"Uknown backend passed: {backend}")


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
