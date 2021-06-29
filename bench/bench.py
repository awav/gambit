from typing import Callable, Tuple, Union
import click
import numpy as np
import tensorflow as tf
import tensorflow.config.experimental as tf_exp
from dataclasses import dataclass
from time import time
import sys
from pathlib import Path
import pandas as pd
from memory_profiler import memory_usage

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from examples import outerprod as outerprod_example
from examples import kernels as kernels_example

# ./bench.py bench --mem-limit 2GB outerprod --input_size "(1e6, 200)"
# ./bench.py bench --mem-limit 2GB innerprod --input_size "(1e6, 200)"
# ./bench.py bench --mem-limit 2GB kernel --type se --input_size "(1e6, 50)"
# ./bench.py bench --mem-limit 2GB kernel --type se --input_size "(1e6, 50)"

__default_gambit_logs = "./default_gambit_logs"


class FloatType(click.ParamType):
    name = "float_type"

    def convert(self, value, param, ctx):
        options = {"fp32": np.float32, "fp64": np.float64}
        try:
            norm_value = value.lower()
            float_type = options[norm_value]
            return float_type
        except:
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

    def run(self, func: Callable):
        gpu_devices = tf.config.get_visible_devices("gpu")
        dev = gpu_devices[0] if gpu_devices else None

        def run_and_collect_stat(func, dev: Union[str, None]):
            if dev is not None:
                time0 = time()
                _ = func()
                elapsed = time() - time0
                mem = tf_exp.get_memory_info(dev)["peak"]
                return elapsed, mem

            func_tuple = (func, [], {})
            time0 = time()
            mem_info = memory_usage(func_tuple)
            elapsed = time() - time0
            mem = np.max(mem_info)
            return elapsed, mem

        for _ in range(self.warmup):
            func()

        elaps, mems = [], []
        for _ in range(self.repeat):
            elapsed, mem = run_and_collect_stat(func, dev)
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


@click.group()
@click.option("-f", "--float-type", type=FloatType(), default="fp64")
@click.option("-m", "--mem-limit", type=MemoryLimit(), default=None)
@click.option("-s", "--seed", type=int, default=None)
@click.option("-r", "--repeat", type=int, default=1)
@click.option("-w", "--warmup", type=int, default=1)
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("--xla/--no-xla", default=True)
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
@click.option("-i", "--input-shape", type=Shape(), required=True)
@click.pass_context
def outerprod_with_itself(ctx: click.Context, input_shape: Tuple[int]):
    cmd_ctx = ctx.obj
    tensor = tf.random.uniform(input_shape, dtype=cmd_ctx.dtype)
    var = tf.Variable(tensor)

    def fn():
        outerprod_example.outerprod(var, var)

    exec_fn = tf.function(fn, experimental_compile=cmd_ctx.xla)
    cmd_ctx.run(exec_fn)


@main.command()
@click.option("-a", "--a-shape", type=Shape(), required=True)
@click.option("-b", "--b-shape", type=Shape(), required=True)
@click.pass_context
def outerprod(ctx: click.Context, a_shape: Tuple[int], b_shape: Tuple[int]):
    cmd_ctx = ctx.obj
    at = tf.random.uniform(a_shape, dtype=cmd_ctx.dtype)
    bt = tf.random.uniform(b_shape, dtype=cmd_ctx.dtype)
    a_var = tf.Variable(at)
    b_var = tf.Variable(bt)

    def fn():
        outerprod_example.outerprod(a_var, b_var)

    exec_fn = tf.function(fn, experimental_compile=cmd_ctx.xla)
    cmd_ctx.run(exec_fn)


kernel_choice = click.Choice(["se", "linear"])


@main.command()
@click.option("-k", "--kernel-name", type=kernel_choice, required=True)
@click.option("-a", "--a-shape", type=Shape(), required=True)
@click.option("-b", "--b-shape", type=Shape(), default=None)
@click.option("-v", "--vector-shape", type=Shape(), required=True)
@click.pass_context
def kernel_vector_product(
    ctx: click.Context, kernel_name: str, a_shape: Tuple, b_shape: Tuple, vector_shape: Tuple
):
    cmd_ctx = ctx.obj
    dim: int = a_shape[-1]
    kernel = kernels_example.create_kernel(kernel_name, dim)
    at = tf.random.uniform(a_shape, dtype=cmd_ctx.dtype)
    bt = at
    if b_shape is not None:
        bt = tf.random.uniform(b_shape, dtype=cmd_ctx.dtype)
    vt = tf.random.uniform(vector_shape, dtype=cmd_ctx.dtype)

    def fn():
        kernels_example.kernel_vector_product(kernel, at, bt, vt)

    exec_fn = tf.function(fn, experimental_compile=cmd_ctx.xla)
    cmd_ctx.run(exec_fn)


if __name__ == "__main__":
    main()