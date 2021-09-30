import ast
from functools import partial
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple, Dict

import click
import gpflow
from gpflow import mean_functions
import numpy as np
import tensorflow as tf
from barelybiasedgp.models import BBGP
from gpflow.models import CGLB, SGPR
from gpflow.kernels import Kernel, Matern32, SquaredExponential

from barelybiasedgp.monitor import Monitor
from barelybiasedgp.selection import greedy_selection_tf as greedy_selection
from barelybiasedgp.optimize import optimize_with_adam, optimize_with_lbfgs

from data import load_data


Dataset = Tuple
DatasetFn = Callable[[int], Dataset]
Tensor = tf.Tensor
__default_logdir = "./logs-dir-default"


class LogdirPath(click.Path):
    def __init__(self, mkdir: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.mkdir = mkdir

    def convert(self, value, param, ctx):
        logdir: str = super().convert(value, param, ctx)
        logdir_path = Path(logdir).expanduser().resolve()
        if self.mkdir:
            logdir_path.mkdir(parents=True, exist_ok=True)
        return logdir_path


class FloatType(click.ParamType):
    name = "dtype"

    def convert(self, value, param, ctx):
        options = {"fp32": np.float32, "fp64": np.float64}
        try:
            norm_value = value.lower()
            dtype = options[norm_value]
            return dtype
        except Exception as ex:
            self.fail(f"{value} is not a valid float type [fp32, fp64]", param, ctx)


class DatasetType(click.ParamType):
    name = "dataset"
    datasets = [
        "snelson1d",
        "elevators",
        "pol",
        "bike",
        "houseelectric",
        "3droad",
        "buzz",
        "keggdirected",
        "keggundirected",
        "song",
    ]

    def convert(self, value, param, ctx):
        dataname = value
        if dataname not in self.datasets:
            self.fail(f"{dataname} dataset is not supported", param, ctx)

        def dataset_fn(seed: int):
            nonlocal dataname
            if dataname != "snelson1d":
                full_name = f"Wilson_{dataname}"
                data = load_data(full_name, seed=seed)
            else:
                data = load_data(dataname, seed=seed)
            return data

        try:
            _dummy = dataset_fn(0)
            return dataset_fn
        except Exception as ex:
            self.fail(f"{dataname} dataset is not supported", param, ctx)


def parse_args(source, keymap):
    params = [kv.split("=") for kv in source.split("_")]
    params = {keymap[k]: ast.literal_eval(v) for k, v in params}
    return params


class KernelType(click.ParamType):
    name = "kernel"
    kernels = {"se": SquaredExponential, "mt32": Matern32}
    param_keymap = {"var": "variance", "len": "lengthscales"}

    @classmethod
    def parse_kernel_parameters(cls, source: str):
        if source is None:
            return {}
        return parse_args(source, cls.param_keymap)

    def convert(self, value, param, ctx):
        try:
            kernel_name, *conf = value.split("_", maxsplit=1)
            source = conf[0] if conf else None
            kernel_class = self.kernels[kernel_name]
            kernel_params = self.parse_kernel_parameters(source)

            def create_kernel_fn(ndim: int):
                positive = gpflow.utilities.positive(1e-5)
                lengthscale = np.ones(ndim)
                if "lengthscales" in kernel_params:
                    lengthscale_param = kernel_params["lengthscales"]
                    lengthscale = lengthscale * lengthscale_param
                kernel = kernel_class()
                kernel.lengthscales = gpflow.Parameter(lengthscale, transform=positive)
                return kernel

            return create_kernel_fn
        except Exception as ex:
            self.fail(f"{value} is not supported", param, ctx)


class ModelType(click.ParamType):
    name = "model"
    models = {"bbgp": BBGP, "cglb": CGLB, "sgpr": SGPR}
    param_keymap = {"m": "num_inducing_points", "bias": "max_bias", "probes": "num_probes"}

    @classmethod
    def parse_model_parameters(cls, source) -> Dict:
        if source is None:
            return {}

        keymap = cls.param_keymap
        model_args = parse_args(source, keymap)
        if model_args:

            def convert(key, value):
                if key == "num_inducing_points" or key == "num_probes":
                    return int(value)
                elif key == "max_bias":
                    return float(value)

            ret = {k: convert(k, v) for k, v in model_args.items()}
            return ret
        return {}

    def convert(self, value, param, ctx):
        try:
            model_name, *conf = value.split("_", maxsplit=1)
            source = conf[0] if conf else None
            model_class = self.models[model_name]
            model_args = self.parse_model_parameters(source)
            num_ip_key = "num_inducing_points"

            def model_fn(
                train_data,
                kernel: Kernel,
                sigma_sq: float,
            ):
                mean = gpflow.mean_functions.Constant()
                if model_name == "bbgp":
                    model = model_class(train_data, kernel=kernel, mean_function=mean, **model_args)
                elif num_ip_key in model_args:
                    num_ip = model_args[num_ip_key]
                    iv, _ = greedy_selection(train_data[0], num_ip, kernel, sigma_sq)
                    if num_ip != iv.shape[0]:
                        raise RuntimeError("Not enough inducing points")
                    model = model_class(
                        train_data, kernel=kernel, inducing_variable=iv, mean_function=mean
                    )

                model.likelihood.variance.assign(sigma_sq)
                return model

            return model_fn
        except Exception as ex:
            self.fail(f"{value} is not supported", param, ctx)


@dataclass
class MainContext:
    dataset: Dataset
    kernel_fn: Callable
    model_fn: Callable
    seed: int
    dtype: np.dtype
    monitor: Monitor


@dataclass
class TrainContext:
    maxiter: int
    jit: bool


@click.group()
@click.option("-d", "--dataset", type=DatasetType())
@click.option("-k", "--kernel", type=KernelType(), default="se")
@click.option("-m", "--model", type=ModelType(), default="bbgp")
@click.option("-s", "--seed", type=int, default=0)
@click.option("-t", "--dtype", type=FloatType(), default="fp64")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_logdir)
@click.pass_context
def main(
    ctx: click.Context,
    dtype: np.dtype,
    logdir: Path,
    kernel: Callable,
    model: Callable,
    seed: int,
    dataset: DatasetFn,
):
    gpflow.config.set_default_float(dtype)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    monitor = Monitor(logdir)
    main_ctx = MainContext(
        dataset(seed),
        kernel,
        model,
        seed,
        dtype,
        monitor,
    )
    ctx.obj = {"main": main_ctx}


@main.group()
@click.option("-i", "--maxiter", type=int)
@click.option("--jit/--no-jit", type=bool, default=True)
@click.pass_context
def train(ctx: click.Context, maxiter: int, jit: bool):
    train_ctx = TrainContext(maxiter=maxiter, jit=jit)
    ctx.obj["train"] = train_ctx


def cast_dataset_to_tensors(data, dtype):
    (x, y), (xt, yt) = data
    ctt = lambda i: tf.convert_to_tensor(i, dtype=dtype)
    data = (ctt(x), ctt(y)), (ctt(xt), ctt(yt))
    return data


def assemble_model(main_ctx: MainContext):
    dataset = cast_dataset_to_tensors(main_ctx.dataset, main_ctx.dtype)
    train_data, _ = dataset
    ndim = train_data[0].shape[-1]
    kernel = main_ctx.kernel_fn(ndim)
    sigma_sq = tf.cast(1.0, main_ctx.dtype)
    model = main_ctx.model_fn(train_data, kernel, sigma_sq)
    return model, dataset


@train.command()
@click.pass_context
def lbfgs(ctx: click.Context):
    main_ctx = ctx.obj["main"]
    train_ctx = ctx.obj["train"]
    model, data = assemble_model(main_ctx)
    monitor = main_ctx.monitor
    jit = train_ctx.jit
    optimize_with_lbfgs(model, data, train_ctx.maxiter, monitor=monitor, jit=jit)
    monitor.close()


@train.command()
@click.option("-l", "--learning-rate", type=float)
@click.pass_context
def adam(ctx: click.Context, learning_rate: float):
    main_ctx = ctx.obj["main"]
    train_ctx = ctx.obj["train"]
    model, data = assemble_model(main_ctx)
    monitor = main_ctx.monitor
    jit = train_ctx.jit
    optimize_with_adam(
        model, data, train_ctx.maxiter, monitor=monitor, learning_rate=learning_rate, jit=jit
    )
    monitor.close()


if __name__ == "__main__":
    main()
