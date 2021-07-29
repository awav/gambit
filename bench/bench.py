from typing import Callable, Tuple, Union
import click
import numpy as np
import tensorflow as tf

_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

if _gpu_dev is not None:
    tf.config.experimental.set_memory_growth(_gpu_dev, True)


import tensorflow.config.experimental as tf_exp
from dataclasses import dataclass
from time import time
import sys
from pathlib import Path
from memory_profiler import memory_usage

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from cases import outerprod as outerprod_example
from cases import kernels as kernels_example
from cases import tril_solve as tril_solve_example
from cases import models as models_example


__default_gambit_logs = "./default_gambit_logs"


class FloatType(click.ParamType):
    name = "float_type"

    def convert(self, value, param, ctx):
        options = {"fp32": np.float32, "fp64": np.float64}
        print(value, param, ctx)
        try:
            norm_value = value.lower()
            float_type = options[norm_value]
            return float_type
        except Exception as ex:
            self.fail(f"{value} is not a valid float type [fp32, fp64]", param, ctx)

    def __repr__(self):
        return "FloatType"


class MemoryLimit(click.ParamType):
    name = "memory_limit"

    def convert(self, value, param, ctx):
        if value is None:
            return value

        options = {"mb": 1, "gb": 1024}
        suffixes = tuple(options.keys())
        try:
            if value.lower().endswith(suffixes):
                return int(value)
            return int(value)
        except:
            self.fail(f"{value} is not a valid float type (allowed fp32, and fp64)", param, ctx)

    def __repr__(self):
        return "MemoryLimit"


class Shape(click.ParamType):
    name = "shape"

    def convert(self, value, param, ctx):
        try:
            values = value.lstrip(" (").rstrip(") ").split(",")
            values = [int(float(v)) for v in values]
            return tuple(values)
        except ValueError:
            self.fail(f"{value} is not in valid shape format", param, ctx)

    def __repr__(self):
        return "FloatType"


class LogdirPath(click.Path):
    def __init__(self, mkdir: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.mkdir = mkdir

    def convert(self, value, param, ctx):
        logdir = super().convert(value, param, ctx)
        logdir_path = Path(logdir).expanduser().resolve()
        if self.mkdir:
            logdir_path.mkdir(parents=True, exist_ok=True)
        return logdir_path


@dataclass
class CommandContext:
    dtype: np.dtype
    memlimit: int
    seed: int
    repeat: int
    warmup: int
    logdir: str
    xla: bool

    def run(self, func_to_run: Callable):
        fn_compiled = tf.function(func_to_run, experimental_compile=self.xla)

        def exec_fn():
            res = fn_compiled()
            if isinstance(res, (list, tuple)):
                return [r.numpy() for r in res]
            return res.numpy()

        gpu_devices = tf.config.get_visible_devices("GPU")
        dev = gpu_devices[0] if gpu_devices else None

        def run_and_collect_stat(func, dev: Union[str, None]):
            if dev is not None:
                time0 = time()
                _ = func()
                elapsed = time() - time0
                dev_name = dev.name.split(":", 1)[-1]
                mem = tf_exp.get_memory_info(dev_name)["peak"]
                return elapsed, mem

            func_tuple = (func, [], {})
            time0 = time()
            mem_info = memory_usage(func_tuple)
            elapsed = time() - time0
            mem = np.max(mem_info)
            return elapsed, mem

        for _ in range(self.warmup):
            exec_fn()

        elaps, mems = [], []
        for _ in range(self.repeat):
            elapsed, mem = run_and_collect_stat(exec_fn, dev)
            elaps.append(elapsed)
            mems.append(mem)

        elaps = np.array(elaps)
        mems = np.array(mems)

        elaps_avg = np.mean(elaps)
        elaps_std = np.std(elaps)
        mem_avg = np.mean(mems)
        mem_std = np.std(mems)

        print(
            "[Bench] Total stat, "
            f"spent_avg={elaps_avg}, spent_std={elaps_std}, "
            f"mem_avg={mem_avg}, mem_std={mem_std}"
        )

        log_file = str(Path(self.logdir, "stats.npy"))
        np.save(
            log_file,
            {
                "elaps": elaps,
                "elaps_avg": elaps_avg,
                "elaps_std": elaps_std,
                "mems": mems,
                "mem_avg": mem_avg,
                "mem_std": mem_std,
            },
        )


_help_doesntwork = "Does not work at the moment. No action will be applied"


@click.group()
@click.option("-f", "--float-type", type=FloatType(), default="fp64")
@click.option("-m", "--mem-limit", type=MemoryLimit(), default=None, help=_help_doesntwork)
@click.option("-s", "--seed", type=int, default=None)
@click.option("-r", "--repeat", type=int, default=1, help="Number of experiment repeatitions")
@click.option("-w", "--warmup", type=int, default=1, help="Number of warm-up iterations")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("--xla/--no-xla", default=True, help="Compile function with or without XLA")
@click.pass_context
def main(
    ctx: click.Context,
    float_type: np.dtype,
    mem_limit: int,
    repeat: int,
    warmup: int,
    logdir: str,
    seed: int,
    xla: bool,
):
    cmd_ctx = CommandContext(
        dtype=float_type,
        memlimit=mem_limit,
        seed=seed,
        repeat=repeat,
        warmup=warmup,
        logdir=logdir,
        xla=xla,
    )
    ctx.obj = cmd_ctx


@main.command()
@click.option("-a", "--a-shape", type=Shape(), required=True)
@click.option("-b", "--b-shape", type=Shape(), default=None)
@click.pass_context
def outerprod(ctx: click.Context, a_shape: Tuple[int], b_shape: Union[Tuple[int], None]):
    cmd_ctx = ctx.obj
    at = tf.random.uniform(a_shape, dtype=cmd_ctx.dtype)
    a_var = tf.Variable(at)

    if b_shape is not None:
        bt = tf.random.uniform(b_shape, dtype=cmd_ctx.dtype)
        b_var = tf.Variable(bt)
    else:
        b_var = a_var

    def fn():
        outerprod_example.outerprod(a_var, b_var)

    cmd_ctx.run(fn)


kernel_choice = click.Choice(["se", "matern32", "linear"])


@main.command()
@click.option("-k", "--kernel-name", type=kernel_choice, required=True)
@click.option("-a", "--a-shape", type=Shape(), required=True)
@click.option("-b", "--b-shape", type=Shape(), default=None)
@click.option("-v", "--vector-shape", type=Shape(), required=True)
@click.pass_context
def kvp(
    ctx: click.Context, kernel_name: str, a_shape: Tuple, b_shape: Tuple, vector_shape: Tuple
):
    cmd_ctx = ctx.obj
    dim: int = a_shape[-1]
    dtype = cmd_ctx.dtype

    kernel = kernels_example.create_kernel(kernel_name, dim, dtype=dtype)

    at = tf.random.uniform(a_shape, dtype=cmd_ctx.dtype)
    bt = None
    if b_shape is not None:
        bt = tf.random.uniform(b_shape, dtype=dtype)

    vt = tf.random.uniform(vector_shape, dtype=dtype)

    def fn():
        return kernels_example.kernel_vector_product(kernel, at, bt, vt)

    cmd_ctx.run(fn)


@main.command()
@click.option("-k", "--kernel-name", type=kernel_choice, required=True)
@click.option("-a", "--a-shape", type=Shape(), required=True)
@click.option("-b", "--b-shape", type=Shape(), default=None)
@click.option("-v", "--vector-shape", type=Shape(), required=True)
@click.pass_context
def kvp_grads(
    ctx: click.Context, kernel_name: str, a_shape: Tuple, b_shape: Tuple, vector_shape: Tuple
):
    cmd_ctx = ctx.obj
    dim: int = a_shape[-1]
    dtype = cmd_ctx.dtype

    kernel = kernels_example.create_kernel(kernel_name, dim, dtype=dtype)

    at = tf.random.uniform(a_shape, dtype=cmd_ctx.dtype)
    bt = None
    if b_shape is not None:
        bt = tf.random.uniform(b_shape, dtype=dtype)

    vt = tf.random.uniform(vector_shape, dtype=dtype)

    def fn():
        return kernels_example.kernel_vector_product(kernel, at, bt, vt)

    cmd_ctx.run(fn)


@main.command()
@click.option("-k", "--kernel-name", type=kernel_choice, default="se")
@click.option("-m", "--matrix-size", type=int, required=True)
@click.option("-b", "--batch-size", type=int, required=True)
@click.option("-d", "--dim", type=int, required=True)
@click.pass_context
def tril_solve(ctx: click.Context, kernel_name: str, matrix_size: int, batch_size: int, dim: int):
    cmd_ctx = ctx.obj
    dtype = cmd_ctx.dtype

    kernel = kernels_example.create_kernel(kernel_name, dim, dtype=dtype)

    at = tf.random.uniform((matrix_size, dim), dtype=dtype)
    bt = tf.random.uniform((batch_size, dim), dtype=dtype)
    matrix = np.random.rand(matrix_size, matrix_size)
    matrix = np.tril(matrix)
    matrix = tf.convert_to_tensor(matrix, dtype=dtype)

    def fn():
        m = matrix
        x = at
        y = bt
        return tril_solve_example.triangular_solve(m, kernel, x, y)

    cmd_ctx.run(fn)


datasets = ["elevators", "pol", "houseelectric", "3droad", "buzz", "keggdirected", "keggundirected", "song"]
dataset_choice = click.Choice(datasets)


@main.command()
@click.option("-k", "--kernel-name", type=kernel_choice, default="se")
@click.option("-d", "--dataset-name", type=dataset_choice, default="elevators")
@click.option("-m", "--num-inducing-points", type=int, default=1000)
@click.option("--grad/--no-grad", type=bool, default=True)
@click.pass_context
def sgpr(ctx: click.Context, kernel_name: str, dataset_name: str, num_inducing_points: int, grad: bool):
    cmd_ctx = ctx.obj
    dtype = cmd_ctx.dtype
    dataset_name = f"Wilson_{dataset_name}"
    fn = models_example.create_sgpr_loss_and_grad(dataset_name, kernel_name, num_inducing_points, do_grad=grad)
    cmd_ctx.run(fn)


if __name__ == "__main__":
    main()