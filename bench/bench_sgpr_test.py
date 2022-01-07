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
from bench_utils import tf_data_tuple, get_uci_dataset
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


# XLA_FLAGS="--xla_try_split_tensor_size=100MB" python bench_sgpr_test.py -s 0 -m 1000


@click.command()
@click.option("-s", "--seed", type=int, default=0)
@click.option("-d", "--dataset-name", type=str, default="keggundirected")
@click.option("-m", "--numips", type=int, help="Number of inducing points")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("--jit/--no-jit", default=True, help="Compile function with or without XLA")
def main(
    dataset_name: int,
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
    data, data_test = get_uci_dataset(dataset_name)
    x, y = data
    data = x, y
    num_data = x.shape[0]
    dim_data = x.shape[-1]

    tf_data = tf_data_tuple(data)
    tf_data_test = tf_data_tuple(data_test)

    noise = 0.1
    max_subset = 10000

    lengthscale = [1.0] * dim_data
    kernel = gpflow.kernels.Matern32(lengthscales=lengthscale)

    iv_phase_1, _ = uniform_greedy_selection(x, max_subset, numips, kernel, noise, 0.001, rng=rng_phase_1)
    model = gpflow.models.SGPR(tf_data, kernel=kernel, inducing_variable=iv_phase_1, noise_variance=noise)
    set_trainable(model.inducing_variable, False)

    vars_phase_1 = model.trainable_variables
    # loss_fn_phase_1 = model.training_loss_closure(compile=False)
    # variable = model.likelihood.variance.unconstrained_variable

    # def test_func():
    #     x, _ = model.data
    #     iv = model.inducing_variable
    #     sigma_sq = model.likelihood.variance

    #     kuf = kufs.Kuf(iv, model.kernel, x)
    #     kuu = kuus.Kuu(iv, model.kernel, jitter=default_jitter())
    #     L = tf.linalg.cholesky(kuu)
    #     sigma = tf.sqrt(sigma_sq)

    #     # Compute intermediate matrices
    #     A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
    #     # A = tf.linalg.triangular_solve(L, kuf, lower=True)
    #     trace_q = tf.reduce_sum(A)

    #     # TODO
    #     # AAT = tf.linalg.matmul(A, A, transpose_b=True)
    #     # trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))

    #     return trace_q

    def test_func():
        x, _ = model.data
        iv = model.inducing_variable
        sigma_sq = model.likelihood.variance

        kuf = kufs.Kuf(iv, model.kernel, x)
        kuu = kuus.Kuu(iv, model.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        sigma = tf.sqrt(sigma_sq)

        # Compute intermediate matrices
        A = kuf / sigma
        A = tf.linalg.triangular_solve(L, kuf, lower=True)
        trace_q = tf.reduce_sum(A)

        # TODO
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))

        return trace_q
    
    # def test_func():
    #     x, _ = model.data
    #     iv = model.inducing_variable
    #     sigma_sq = model.likelihood.variance

    #     kuf = kufs.Kuf(iv, model.kernel, x)
    #     # kuu = kuus.Kuu(iv, model.kernel, jitter=default_jitter())
    #     # L = tf.linalg.cholesky(kuu)
    #     sigma = tf.sqrt(sigma_sq)

    #     # Compute intermediate matrices
    #     A = kuf / sigma
    #     # A = tf.linalg.triangular_solve(L, kuf, lower=True)
    #     trace_q = tf.reduce_sum(A)

    #     # TODO
    #     # AAT = tf.linalg.matmul(A, A, transpose_b=True)
    #     # trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))

    #     return trace_q

    loss_fn_phase_1 = test_func

    # opt = Scipy()
    # res = opt.minimize(loss_fn_phase_1, vars_phase_1, compile=xla_flag, options=dict(maxiter=maxiter))
    # print(res)

    def loss_grad_fn():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch([vars_phase_1])
            loss = loss_fn_phase_1()
        grads = tape.gradient(loss, vars_phase_1)
        return loss, grads

    # def loss_grad_fn():
    #     loss = loss_fn_phase_1()
    #     return loss

    # loss_fn_phase_1_jit = tf.function(loss_fn_phase_1, jit_compile=jit)
    # elbo = loss_fn_phase_1_jit()

    loss_fn_phase_1_jit = tf.function(loss_grad_fn, jit_compile=jit)

    # elbo, grad = loss_fn_phase_1_jit()
    # print(f"ELBO: {elbo.numpy()}, kernel variance gradient: {grad.numpy()}")

    values = loss_fn_phase_1_jit()
    if not isinstance(values, (list, tuple)):
        values = [values]

    numpy_values = []
    for v in values:
        if isinstance(v, (list, tuple)):
            acc = [(v_ if v_ is None else v_.numpy()) for v_ in v]
            numpy_values.append(acc)
        else:
            numpy_values.append(v.numpy())

    print(f"Values: {numpy_values}")


if __name__ == "__main__":
    main()
