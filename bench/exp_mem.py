import sys
import json
from pathlib import Path
from typing import Callable, Tuple, NamedTuple
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import gpflow

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import LogdirPath
from bench_utils import BenchRunner, store_dict_as_h5
from bench_sgpr_utils import (
    compile_function,
    CompileType
)

__default_gambit_logs = "./logs_mem_default"
__gpu_devices = tf.config.get_visible_devices("GPU")
__gpu_dev = __gpu_devices[0] if __gpu_devices else None


if __gpu_dev is not None:
    click.echo(f">>> GPU device information: {__gpu_dev}")
    click.echo(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(__gpu_dev, True)


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-m", "--memory-limit", type=str)
@click.option("-n", "--size", type=int)
@click.option("-s", "--seed", type=int, default=0)
@click.option("-r", "--repeat", type=int, default=1)
@click.option("-w", "--warmup", type=int, default=1)
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
def main(
    size: int,
    memory_limit: int,
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    compile: Literal["xla", "tf", "none"],
):
    info = {
        "size": size,
        "memory_limit": memory_limit,
        "seed": seed,
        "repeat": repeat,
        "warmup": warmup,
        "compile": compile,
    }
    info_str = json.dumps(info, indent=2)
    print("===> Starting")
    print(f"-> {info_str}")
    assert Path(logdir).exists()

    compile_flag: CompileType = compile if compile != "none" else None

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpflow_dtype = gpflow.config.default_float()

    def ctt(x, dtype=None):
        dtype = gpflow_dtype if dtype is None else dtype
        return tf.convert_to_tensor(x, dtype=dtype)

    x = rng.randn(size, 1)
    y = rng.randn(size, 1)
    v = rng.randn(size, 1)

    x_tf = ctt(x)
    y_tf = ctt(y)
    v_tf = ctt(v)

    kernel = gpflow.kernels.SquaredExponential()

    def eval_test(x_data, y_data, vec):
        k = kernel(x_data, y_data)
        return k @ vec

    eval_test_compiled = compile_function(eval_test, compile_flag)

    bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
    results = bench_runner.bench(eval_test_compiled, [x_tf, y_tf, v_tf])
    bench_table = {**info, **results}

    filepath = Path(logdir, "bench.h5")
    store_dict_as_h5(bench_table, filepath)

    (elap_mu, elap_std) = results["elapsed_stats"]
    (mem_mu, mem_std) = results["mem_stats"]

    print(
        "[Bench] Total stat, "
        f"spent_avg={elap_mu}, spent_std={elap_std}, "
        f"mem_avg={mem_mu}, mem_std={mem_std}"
    )


if __name__ == "__main__":
    main()
