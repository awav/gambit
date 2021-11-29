import sys
from pathlib import Path
from typing import Dict, Tuple, NamedTuple
import click
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.config import default_jitter
from gpflow.utilities import set_trainable
from gpflow.covariances import kufs, kuus

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import FloatType, LogdirPath
from cases.models import get_dataset, to_tf_data
from barelybiasedgp.selection import uniform_greedy_selection
from barelybiasedgp.scipy_copy import Scipy

Dataset = Tuple[np.ndarray, np.ndarray]
DatasetBundle = NamedTuple


_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

# if _gpu_dev is not None:
#     print(f">>> GPU device information: {_gpu_dev}")
#     print(">>> Set GPU memory growth")
#     tf.config.experimental.set_memory_growth(_gpu_dev, True)

__default_gambit_logs = "./default_gambit_logs"
__datasets = click.Choice(["houseelectric", "song", "buzz", "3droad", "keggundirected"])


# XLA_FLAGS="--xla_try_split_tensor_size=100MB --xla_enable_hlo_passes_only=split-intermediate-tensors,broadcast-simplifier,dce,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-test-100m" python bench_matmul.py -s 999 -n 1000753 -d 1000

# XLA_FLAGS="--xla_try_split_tensor_size=100MB" python bench_matmul.py -s 999 -n 1000753 -d 1000
# XLA_FLAGS="--xla_try_split_tensor_size=10GB" python bench_matmul.py -s 999 -n 1000753 -d 1000

# XLA_FLAGS="--xla_try_split_tensor_size=100MB" python bench_matmul.py -s 999 -n 1000753 -d 1000
# XLA_FLAGS="--xla_try_split_tensor_size=10GB" python bench_matmul.py -s 999 -n 1000753 -d 1000



@click.command()
@click.option("-s", "--seed", type=int, default=0)
@click.option("-d", "--dataset-name", type=str, default="keggundirected")
@click.option("-n", "--maxiter", type=int, default=100)
@click.option("-m", "--numips", type=int, help="Number of inducing points")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("--jit/--no-jit", default=True, help="Compile function with or without XLA")
def main(
    dataset_name: int,
    maxiter: int,
    numips: int,
    logdir: str,
    seed: int,
    jit: bool,
):
    assert Path(logdir).exists()
    rng_phase_1 = np.random.RandomState(seed)
    rng_phase_2 = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    xla_flag = "xla" if jit else None
    data, data_test = get_dataset(f"Wilson_{dataset_name}")
    x, y = data
    # x, y = x[:37500], y[:37500]
    data = x, y
    num_data = x.shape[0]
    dim_data = x.shape[-1]

    tf_data = to_tf_data(data)
    tf_data_test = to_tf_data(data_test)

    noise = 0.1
    max_subset = 100000

    lengthscale = [1.0] * dim_data
    kernel = gpflow.kernels.Matern32(lengthscales=lengthscale)

    iv_phase_1, _ = uniform_greedy_selection(x, max_subset, numips, kernel, noise, 0.001, rng=rng_phase_1)
    model = gpflow.models.SGPR(tf_data, kernel=kernel, inducing_variable=iv_phase_1, noise_variance=noise)
    set_trainable(model.inducing_variable, False)

    vars_phase_1 = model.trainable_variables
    loss_phase_1 = model.training_loss_closure(compile=False)

    print(">>> Phase 1")
    opt = Scipy()
    res = opt.minimize(loss_phase_1, vars_phase_1, compile=xla_flag, options=dict(maxiter=maxiter))
    print(res)

    metrics_phase_1 = make_metrics_func(model, data_test)

    loss_phase_1_jit = tf.function(loss_phase_1, jit_compile=jit)
    metrics_phase_1_jit = tf.function(metrics_phase_1, jit_compile=jit)

    elbo = loss_phase_1_jit().numpy()
    metric = metrics_phase_1_jit()
    metric = {k: v.numpy() for (k, v) in metric.items()}

    print(f"ELBO after phase 1: {elbo}")
    print(f"Metrics after phase 1: {metric}")

    # Phase 2
    print(">>> Phase 2")
    iv_phase_2, _ = uniform_greedy_selection(x, max_subset, numips, kernel, noise, 0.001, rng=rng_phase_2)
    model.inducing_variable.Z.assign(iv_phase_2)

    set_trainable(model.inducing_variable, True)
    vars_phase_2 = model.trainable_variables
    loss_phase_2 = model.training_loss_closure(compile=False)
    metrics_phase_2 = make_metrics_func(model, data_test)
    metrics_phase_2_jit = tf.function(metrics_phase_2, jit_compile=jit)

    max_attempts = 4
    for i in range(max_attempts):
        print(f">>> Optimizing attempt: [{i}/{max_attempts}]\n")
        opt = Scipy()
        res = opt.minimize(loss_phase_2, vars_phase_2, compile=xla_flag, options=dict(maxiter=maxiter))
        print(res)
        metric = metrics_phase_2_jit()
        metric = {k: v.numpy() for (k, v) in metric.items()}
        print(f"Metrics after phase 2: {metric}")

        if res.nit < 2:
            break


def make_metrics_func(model, data):
    def _metrics():
        x, y = data
        predictions, variance = model.predict_f(x)
        err = y - predictions
        logden = model.likelihood.predict_log_density(predictions, variance, y)
        err_sq = tf.square(err)
        rmse = tf.sqrt(tf.reduce_mean(err_sq))
        nlpd = -tf.reduce_mean(logden)
        return dict(rmse=rmse, nlpd=nlpd)
    return _metrics


if __name__ == "__main__":
    main()
