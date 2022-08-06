import sys
import json
from pathlib import Path
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import gpflow

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from transformer import Transformer
from clitypes import LogdirPath
from bench_utils import BenchRunner, store_dict_as_h5
from bench_sgpr_utils import (
    compile_function,
    CompileType
)

__default_gambit_logs = "./logs_transformer_default"
__gpu_devices = tf.config.get_visible_devices("GPU")
__gpu_dev = __gpu_devices[0] if __gpu_devices else None


if __gpu_dev is not None:
    click.echo(f">>> GPU device information: {__gpu_dev}")
    click.echo(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(__gpu_dev, True)

# New version after renaming
# TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-exp-transformer" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=10GB --xla_tensor_split_size=1GB --xla_enable_hlo_passes_only=tensor-splitter,algebraic-rewriter,dce,broadcast-simplifier,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander,CallInliner,gpu_scatter_expander,rce-optimizer" python ./exp_transformer.py --sequence-len 10000 2>&1 | tee output-exp-transformer.log

# New version after renaming
# TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-exp-transformer" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=20GB --xla_tensor_split_size=10GB" python ./exp_transformer.py --sequence-len 10000 2>&1 | tee output-exp-transformer.log


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-m", "--memory-limit", type=str)
@click.option("-sl", "--sequence-len", type=int)
@click.option("-s", "--seed", type=int, default=0)
@click.option("-r", "--repeat", type=int, default=1)
@click.option("-w", "--warmup", type=int, default=1)
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
def main(
    sequence_len: int,
    memory_limit: int,
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    compile: Literal["xla", "tf", "none"],
):
    memory_limit = "none" if memory_limit is None else memory_limit
    info = {
        "sequence_len": sequence_len,
        "memory_limit": memory_limit,
        "seed": seed,
        "repeat": repeat,
        "warmup": warmup,
        "compile": compile,
    }
    info_str = json.dumps(info, indent=2)
    click.echo("===> Starting")
    click.echo(f"-> {info_str}")
    assert Path(logdir).exists()

    compile_flag: CompileType = compile if compile != "none" else None

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpflow_dtype = gpflow.config.default_float()

    def ctt(x, dtype=None):
        dtype = gpflow_dtype if dtype is None else dtype
        return tf.convert_to_tensor(x, dtype=dtype)

    batch_size = 64
    d_model = 512
    num_heads = 8
    input_shape = (batch_size, sequence_len)

    # Max size is prod([batch_size, num_heads, seq_len, seq_len, d_model]) * precision

    # q = rng.randn(*input_shape)
    # k = rng.randn(*input_shape)
    # v = rng.randn(*input_shape)
    # q_tf = ctt(q)
    # k_tf = ctt(k)
    # v_tf = ctt(v)

    x = rng.uniform(0, 200, size=input_shape)
    y = rng.uniform(0, 200, size=input_shape)
    x_tf = ctt(x)
    y_tf = ctt(y)

    transformer = Transformer(
        num_layers=2, d_model=d_model, num_heads=num_heads, dff=2048,
        input_vocab_size=8500, target_vocab_size=8000)

    def eval_test(inputs, targets):
        out =  transformer([inputs, targets], training=False)
        return out

    eval_test_compiled = compile_function(eval_test, compile_flag)

    bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
    results = bench_runner.bench(eval_test_compiled, [x_tf, y_tf])
    bench_table = {**info, **results}

    filepath = Path(logdir, "bench.h5")
    store_dict_as_h5(bench_table, filepath)

    if "elapsed_stats" not in results or "mem_stats" not in results:
        click.echo("⚠️ No stats in the benchmark output ⚠️ ")
        raise click.exceptions.Exit(0)

    (elap_mu, elap_std) = results["elapsed_stats"]
    (mem_mu, mem_std) = results["mem_stats"]

    click.echo(
        "[Bench] Total stat, "
        f"spent_avg={elap_mu}, spent_std={elap_std}, "
        f"mem_avg={mem_mu}, mem_std={mem_std}"
    )

if __name__ == "__main__":
    main()
