from typing import Callable, Tuple, Union
import click
import numpy as np
import tensorflow as tf

_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

if _gpu_dev is not None:
    print(f">>> GPU device information: {_gpu_dev}")
    print(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(_gpu_dev, True)


import tensorflow.config.experimental as tf_exp
from dataclasses import dataclass
from time import time
import sys
from pathlib import Path
from memory_profiler import memory_usage

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from cases import backends as backends_case
from cases import dists as dists_example
from cases import outerprod as outerprod_example
from cases import matrix_chain as matrix_chain_example
from cases import kernels as kernels_example
from cases import tril_solve as tril_solve_example
from cases import models as models_example
from clitypes import FloatType, LogdirPath, Shape


__default_gambit_logs = "./default_gambit_logs"

@dataclass
class CommandContext:
    dtype: np.dtype
    seed: int
    repeat: int
    warmup: int
    logdir: str
    xla: bool
    backend: str

    def run(self, func_to_run: Callable):
        fn_compiled = backends_case.jit_compile(self.backend, func_to_run)
        # fn_compiled = func_to_run
        sync_cpu = lambda value: backends_case.sync_cpu(self.backend, value)

        def exec_fn():
            res = fn_compiled()
            if isinstance(res, (list, tuple)):
                return [sync_cpu(r) for r in res]
            return sync_cpu(res)

        gpu_devices = tf.config.get_visible_devices("GPU")
        dev = gpu_devices[0] if gpu_devices else None

        def run_and_collect_stat(func, dev: Union[str, None]):
            if dev is not None:
                dev_name = dev.name.split(":", 1)[-1]
                time0 = time()
                tf_exp.reset_memory_stats(dev_name)
                _ = func()
                elapsed = time() - time0
                mem_peak = tf_exp.get_memory_info(dev_name)["peak"]
                # mem_curr = tf_exp.get_memory_info(dev_name)["current"]
                return elapsed, mem_peak

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


frameworks = click.Choice(["tf", "jax"])


@click.group()
@click.option("-f", "--float-type", type=FloatType(), default="fp64")
@click.option("-s", "--seed", type=int, default=None)
@click.option("-r", "--repeat", type=int, default=1, help="Number of experiment repeatitions")
@click.option("-w", "--warmup", type=int, default=1, help="Number of warm-up iterations")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("--xla/--no-xla", default=True, help="Compile function with or without XLA")
@click.option("-b", "--backend", default="tf", type=frameworks, help="TensorFlow or JAX framework")
@click.pass_context
def main(
    ctx: click.Context,
    float_type: np.dtype,
    repeat: int,
    warmup: int,
    logdir: str,
    seed: int,
    xla: bool,
    backend: str,
):
    cmd_ctx = CommandContext(
        dtype=float_type,
        seed=seed,
        repeat=repeat,
        warmup=warmup,
        logdir=logdir,
        xla=xla,
        backend=backend,
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


chain_choice = click.Choice(["reduce_sum_chain", "transpose_chain", "pure_matrix_chain", "matrix_vector_chain"])

@main.command()
@click.option("-n", "--chain_type", type=chain_choice, default="reduce_sum_chain", help="The type of matrix chain")
@click.option("-n", "--digit_num", type=int, default=1000, help="The number of digits of each size of matrix dimensions")
@click.pass_context
def matrix_chain(ctx: click.Context, chain_type: str, digit_num: int):
    cmd_ctx = ctx.obj
    if digit_num<10:
        digit_num=10
    A = tf.Variable(tf.random.uniform((4*digit_num, 2*digit_num)))
    B = tf.Variable(tf.random.uniform((2*digit_num, 3*digit_num))) 
    C = tf.Variable(tf.random.uniform((3*digit_num, 1*digit_num))) 
    D = tf.Variable(tf.random.uniform((1*digit_num, ))) 
    E = tf.Variable(tf.random.uniform((2*digit_num,5*digit_num//10)))
    F = tf.Variable(tf.random.uniform((5*digit_num//10,1*digit_num)))
    fn = lambda : matrix_chain_example.reduce_matrix_chain(A,B,C,D)
    if chain_type == "transpose_chain":
        A = tf.Variable(tf.random.uniform((4*digit_num, 2*digit_num)))
        B = tf.Variable(tf.random.uniform((2*digit_num, 3*digit_num)))
        C = tf.Variable(tf.random.uniform((3*digit_num, 2*digit_num)))
        D = tf.Variable(tf.random.uniform((3*digit_num, 1*digit_num)))
        fn = lambda : matrix_chain_example.transpose_chain(A,B,C,D,E,F)
    elif chain_type == "matrix_vector_chain":
        A = tf.Variable(tf.random.uniform((4*digit_num,2*digit_num)))
        B = tf.Variable(tf.random.uniform((2*digit_num,)))
        C = tf.Variable(tf.random.uniform((4*digit_num, 3*digit_num)))
        D = tf.Variable(tf.random.uniform((1*digit_num, 3*digit_num)))
        fn = lambda : matrix_chain_example.matrix_vector_chain(A,B,C,D)
    elif chain_type == "pure_matrix_chain":
        A = tf.Variable(tf.random.uniform((1*digit_num, 6*digit_num)))
        B = tf.Variable(tf.random.uniform((6*digit_num, 3*digit_num)))
        C = tf.Variable(tf.random.uniform((3*digit_num, 2*digit_num)))
        D = tf.Variable(tf.random.uniform((2*digit_num, 4*digit_num)))
        E = tf.Variable(tf.random.uniform((4*digit_num, 1*digit_num)))
        fn = lambda : matrix_chain_example.pure_matrix_chain(A,B,C,D,E)
    
    cmd_ctx.run(fn)


kernel_choice = click.Choice(["se", "matern32", "linear"])


@main.command()
@click.option("-k", "--kernel-name", type=kernel_choice, required=True)
@click.option("-a", "--a-shape", type=Shape(), required=True)
@click.option("-b", "--b-shape", type=Shape(), default=None)
@click.option("-v", "--vector-shape", type=Shape(), required=True)
@click.pass_context
def kvp(ctx: click.Context, kernel_name: str, a_shape: Tuple, b_shape: Tuple, vector_shape: Tuple):
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
@click.option("-d", "--dim", type=int, required=True)
@click.option("-a", "--a-size", type=int, required=True)
@click.option("-b", "--b-size", type=int, required=None)
@click.pass_context
def dist(ctx: click.Context, dim: int,  a_size: int, b_size: int):
    cmd_ctx = ctx.obj
    fn = dists_example.create_dist_function_evaluation(dim, a_size, b_size, cmd_ctx.seed)
    cmd_ctx.run(fn)


# XLA_FLAGS="--xla_try_split_tensor_size=10GB" python ./bench.py --warmup 10 --repeat 100 --logdir "./logs/kvp/fp64-split_10GB_se-500000-10" -f fp64 -r 10 -w 1 kvp -k se -a "(500000, 10)" -b "(500000, 10)" -v "(500000, 1)"
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
        return kernels_example.kernel_vector_product_with_grads(kernel, at, bt, vt)

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


datasets = [
    "elevators",
    "pol",
    "houseelectric",
    "3droad",
    "buzz",
    "keggdirected",
    "keggundirected",
    "song",
]
dataset_choice = click.Choice(datasets)


@main.command()
@click.option("-k", "--kernel-name", type=kernel_choice, default="se")
@click.option("-d", "--dataset-name", type=dataset_choice, default="elevators")
@click.option("-m", "--num-inducing-points", type=int, default=1000)
@click.option("--grad/--no-grad", type=bool, default=True)
@click.pass_context
def sgpr(
    ctx: click.Context, kernel_name: str, dataset_name: str, num_inducing_points: int, grad: bool
):
    cmd_ctx = ctx.obj
    dtype = cmd_ctx.dtype
    dataset_name = f"Wilson_{dataset_name}"
    fn = models_example.create_sgpr_loss_and_grad(
        dataset_name, kernel_name, num_inducing_points, do_grad=grad
    )
    cmd_ctx.run(fn)


if __name__ == "__main__":
    main()
