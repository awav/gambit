import sys
import os
import json
import pprint
from pathlib import Path
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
from bench.bench_sgpr_utils import compile_function

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import LogdirPath
from bench_utils import get_uci_dataset, tf_data_tuple

from bench_sgpr_utils import (
    make_initialize_ips_function,
    initialize_sgpr,
)

__default_gambit_logs = "./logs_sgpr_default"
DatasetChoices = click.Choice(
    ["houseelectric", "song", "buzz", "3droad", "keggundirected", "protein", "kin40k"]
)

_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

if _gpu_dev is not None:
    click.echo(f">>> GPU device information: {_gpu_dev}")
    click.echo(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(_gpu_dev, True)


# TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-spgr-hlo-ir" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=1GB --xla_tensor_split_size=1GB --xla_enable_hlo_passes_only=tensor-splitter,algebraic-rewriter,dce,broadcast-simplifier,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander,CallInliner,gpu_scatter_expander,rce-optimizer" python ./sgpr_for_hlo_ir.py -l logs/sgpr_hlo_ir_1GB -s 777 -d 3droad -c xla -m 3000 2>&1 | tee logs-sgpr-.log


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-s", "--seed", type=int, default=0)
@click.option("-d", "--dataset-name", type=DatasetChoices, default="3droad")
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
@click.option("-ss", "--subset-size", default=100000, help="Greedy selection subset size")
@click.option("-m", "--numips", type=int, help="Number of inducing points")
def main(
    dataset_name: str,
    numips: int,
    logdir: str,
    seed: int,
    compile: Literal["xla", "tf", "none"],
    subset_size: int,
):
    click.echo("===> Starting")
    assert Path(logdir).exists()

    compile_flag = compile if compile != "none" else None

    data, data_test = get_uci_dataset(dataset_name, seed)
    tf_data = tf_data_tuple(data)
    tf_data_test = tf_data_tuple(data_test)

    info = {
        "seed": seed,
        "dataset_name": dataset_name,
        "numips": numips,
        "compile": compile,
        "subset_size": subset_size,
        "dim_size": data[0].shape[-1],
        "train_size": data[0].shape[0],
        "test_size": data_test[0].shape[0],
    }
    info_str = json.dumps(info, indent=2)
    click.echo(f"-> {info_str}")

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    noise = 0.1
    click.echo("==> Initialize SGPR")
    model = initialize_sgpr(rng, tf_data, numips, noise)

    rng = np.random.RandomState(seed)

    threshold = 1e-6
    initialize_ips_fn = make_initialize_ips_function(
        rng, data, model, numips, subset_size, threshold=threshold
    )
    initialize_ips_fn()

    train_vars = model.trainable_variables
    loss_fn = model.training_loss_closure(compile=False)

    def grad_and_loss_fn():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(train_vars)
            loss = loss_fn()
        grads = tape.gradient(loss, train_vars)
        return loss, grads

    click.echo("==> Run SGPR loss and gradient")
    jit_grad_and_loss_fn = compile_function(grad_and_loss_fn, compile_flag)

    loss, grads = jit_grad_and_loss_fn()
    loss_np = loss.numpy()
    grads_np = [g.numpy() for g in grads]

    click.echo("<=== Finished")


if __name__ == "__main__":
    pprint.pprint(dict(os.environ), width=1)
    main()
