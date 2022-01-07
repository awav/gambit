from functools import partial
import sys
import json
from pathlib import Path
from typing import Callable, Optional, Union, Tuple, NamedTuple, Dict
from typing_extensions import Literal
import click
from gpflow.utilities.traversal import parameter_dict
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.utilities import set_trainable

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import LogdirPath
from monitor import Monitor
from bench_utils import get_uci_dataset, store_dict_as_h5, tf_data_tuple, to_tf_scope
from barelybiasedgp.selection import uniform_greedy_selection
from barelybiasedgp.scipy_copy import Scipy

CompileType = Union[Literal["xla", "tf", "none"], Union[Literal["xla", "tf"], None]]
Dataset = Tuple[np.ndarray, np.ndarray]
DatasetBundle = NamedTuple

__default_gambit_logs = "./logs_sgpr_default"
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
@click.option("-d", "--dataset-name", type=DatasetChoices, default="keggundirected")
@click.option("-np", "--num_phases", type=int, default=1)
@click.option("-ni", "--maxiter_per_phase", type=int, default=100)
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
@click.option("-ss", "--subset-size", default=100000, help="Greedy selection subset size")
@click.option("--monitor-metrics/--no-monitor-metrics", default=False)
@click.option(
    "-hi", "--holdout_interval", default=1, type=int, help="Holdout interval between recordings"
)
@click.option("-m", "--numips", type=int, help="Number of inducing points")
def main(
    dataset_name: str,
    num_phases: int,
    maxiter_per_phase: int,
    numips: int,
    logdir: str,
    seed: int,
    compile: Literal["xla", "tf", "none"],
    subset_size: int,
    holdout_interval: int,
    monitor_metrics: bool,
):
    info = {
        "seed": seed,
        "dataset_name": dataset_name,
        "num_phases": num_phases,
        "maxiter_per_phase": maxiter_per_phase,
        "numips": numips,
        "compile": compile,
        "subset_size": subset_size,
        "monitor_metrics": monitor_metrics,
    }
    info_str = json.dumps(info, indent=2)
    print("===> Starting")
    print(f"-> {info_str}")
    assert Path(logdir).exists()

    compile_flag: CompileType = compile if compile != "none" else None

    threshold = 0.001
    noise = 0.1

    data, data_test = get_uci_dataset(dataset_name, seed)
    tf_data = tf_data_tuple(data)
    tf_data_test = tf_data_tuple(data_test)

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = initialize_sgpr(rng, tf_data, numips, noise)
    set_trainable(model.inducing_variable, False)
    scipy_options = dict(maxiter=maxiter_per_phase)
    loss_fn = model.training_loss_closure(compile=False)

    monitor_param_fn = create_model_parameters_func(model)
    monitor_metric_fn = create_metrics_func(model, tf_data_test, compile_flag, prefix="test")
    param_key = "param"
    metric_key = "metric"
    funcs = {param_key: monitor_param_fn}
    if monitor_metrics:
        funcs[metric_key] = monitor_metric_fn

    monitor = create_monitor(holdout_interval, logdir, funcs=funcs)

    train_vars = model.trainable_variables
    for ph in range(num_phases):
        print(f"---> Phase {ph}")
        rng = np.random.RandomState(seed)
        initialize_ips(rng, data, model, numips, subset_size, threshold)
        current_ips = model.inducing_variable.Z.numpy()
        opt = Scipy()
        res = opt.minimize(
            loss_fn,
            train_vars,
            step_callback=monitor,
            compile=compile_flag,
            options=scipy_options,
        )

        if res.nit < 2:
            print(
                f"---> Break training loop after phase {ph}."
                f"The number of optimization iterations is at minimum: {res.nit}"
            )
            break

        print("-> optimisation result:")
        print(res)

    logs = monitor.collect_logs()
    param_path = Path(logdir, "param.h5")
    store_dict_as_h5(logs[param_key], param_path)
    info_extension = {f"{param_key}_path": str(param_path)}

    if monitor_metrics:
        metric_path = Path(logdir, "metric.h5")
        store_dict_as_h5(logs[metric_key], metric_path)
        info_extension = {f"{param_key}_path": str(metric_path), **info_extension}
    
    final_metric = numpy_results(monitor_metric_fn)
    info_extension = {"final_metric": final_metric, **info_extension}
    info_extended = {**info, **info_extension}
    info_path = Path(logdir, "info.h5")
    store_dict_as_h5(info_extended, info_path)

    print("<=== Finished")


def initialize_ips(
    rng,
    data,
    model,
    numips: int,
    subset_size: int,
    threshold: float,
):
    x, _ = data
    kernel = model.kernel
    noise = np.array(model.likelihood.variance)
    iv, _ = uniform_greedy_selection(x, subset_size, numips, kernel, noise, threshold, rng=rng)
    model.inducing_variable.Z.assign(iv)


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


def create_monitor(
    holdout_interval: int,
    logdir: Path,
    funcs: Optional[Dict[str, Callable]] = None,
):
    funcs_dict: Dict[str, Callable] = {} if funcs is None else funcs
    monitor_logdir = Path(logdir, "tb")
    monitor = Monitor(monitor_logdir, holdout_interval=holdout_interval)

    def cb_wrapper(func: Callable, *args, **kwargs):
        return func()

    for name, func in funcs_dict.items():
        cb_func = partial(cb_wrapper, func)
        monitor.add_callback(name, cb_func)
    return monitor


def numpy_results(metric_fn: Callable):
    def _map(value):
        if isinstance(value, tf.Tensor):
            return value.numpy()
        return np.array(value)

    values = metric_fn()
    return tf.nest.map_structure(_map, values)


def create_metrics_func(model, data, compile_flag: CompileType, prefix: Optional[str] = None):
    def _metrics():
        x, y = data
        predictions, variance = model.predict_f(x)
        err = y - predictions
        logden = model.likelihood.predict_log_density(predictions, variance, y)
        err_sq = tf.square(err)
        rmse = tf.sqrt(tf.reduce_mean(err_sq))
        nlpd = -tf.reduce_mean(logden)
        if prefix is not None:
            return {f"{prefix}_rmse": rmse, f"{prefix}_nlpd": nlpd}
        return {"rmse": rmse, "nlpd": nlpd}

    return compile_function(_metrics, compile_flag)


def create_model_parameters_func(model: gpflow.models.SGPR):
    def _params():
        params = {}
        for name, param in parameter_dict(model).items():
            if not param.trainable:
                continue
            name = name.lstrip(".")
            params[name] = param.numpy()
        return params

    return _params


if __name__ == "__main__":
    main()
