import sys
import json
from pathlib import Path
from typing import Callable, Tuple, NamedTuple
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import set_trainable

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import LogdirPath
from bench_utils import BenchRunner, get_uci_dataset, store_dict_as_h5, tf_data_tuple, to_tf_scope

Dataset = Tuple[np.ndarray, np.ndarray]
DatasetBundle = NamedTuple

__default_gambit_logs = "./logs_sgpr_iter_speed_default"
DatasetChoices = click.Choice(
    ["houseelectric", "song", "buzz", "3droad", "keggundirected", "protein", "kin40k"]
)

_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

if _gpu_dev is not None:
    print(f">>> GPU device information: {_gpu_dev}")
    print(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(_gpu_dev, True)


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-s", "--seed", type=int, default=0)
@click.option("-r", "--repeat", type=int, default=1)
@click.option("-w", "--warmup", type=int, default=1)
@click.option("-d", "--dataset", type=DatasetChoices, default="keggundirected")
@click.option("-m", "--numips", type=int, help="Number of inducing points")
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
@click.option("-ss", "--subset-size", default=100000, help="Greedy selection subset size")
@click.option("--grad-ips/--no-grad-ips", default=False, help="Take the gradient of ips variables")
def main(
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    dataset: str,
    numips: int,
    compile: Literal["xla", "tf", "none"],
    subset_size: int,
    grad_ips: bool,
):
    info = {
        "seed": seed,
        "dataset_name": dataset,
        "repeat": repeat,
        "warmup": warmup,
        "numips": numips,
        "compile": compile,
        "subset_size": subset_size,
        "grad_ips": grad_ips,
    }
    info_str = json.dumps(info, indent=2)
    print("===> Starting")
    print(f"-> {info_str}")
    assert Path(logdir).exists()

    compile_flag: CompileType = compile if compile != "none" else None

    noise = 0.1

    data, _ = get_uci_dataset(dataset, seed)
    tf_data = tf_data_tuple(data)

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = initialize_sgpr(rng, tf_data, numips, noise)
    set_trainable(model.inducing_variable, grad_ips)

    train_vars = model.trainable_variables
    loss_fn = model.training_loss_closure(compile=False)

    def loss_and_grad():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(train_vars)
            loss = loss_fn()
        grad = tape.gradient(loss, train_vars)
        return loss, grad

    loss_and_grad_func = compile_function(loss_and_grad, compile_flag)

    bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
    results = bench_runner.bench(loss_and_grad_func, [])
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


def initialize_sgpr(
    rng,
    data,
    numips: int,
    noise: float,
):
    x, _ = data
    dim = x.shape[-1]
    lengthscale = [1.0] * dim
    kernel = gpflow.kernels.Matern32(lengthscales=lengthscale)
    x = np.array(x)
    dataset_size = x.shape[0]
    subset_indices = rng.choice(dataset_size, size=numips, replace=False)
    subset_mask = np.array([False] * dataset_size)
    subset_mask[subset_indices] = True
    iv = to_tf_scope(x[subset_mask])
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.SGPR(
        data, kernel=kernel, mean_function=mean, inducing_variable=iv, noise_variance=noise
    )
    return model


def compile_function(fn: Callable, compile_type: CompileType) -> Callable:
    if compile_type == "xla":
        return tf.function(fn, jit_compile=True)
    elif compile_type == "tf":
        return tf.function(fn)
    return fn


if __name__ == "__main__":
    main()
