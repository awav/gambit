from pathlib import Path
from typing import Optional, Union, Callable, Dict, NamedTuple, Tuple
from typing_extensions import Literal
from scipy.optimize import OptimizeResult
import gpflow
import numpy as np
import tensorflow as tf

from gpflow.utilities import parameter_dict
from bench_utils import to_tf_scope, OsPath
from monitor import Monitor
from barelybiasedgp.selection import (
    run_uniform_greedy_selection,
    make_inducing_selection_function,
    uniform_greedy_selection,
)


CompileType = Union[Literal["xla", "tf", "none", "tvm"], Union[Literal["xla", "tf", "tvm"], None]]
Dataset = Tuple[np.ndarray, np.ndarray]
DatasetBundle = NamedTuple


def make_initialize_ips_function(
    rng,
    data,
    model,
    numips: int,
    subset_size: int,
    threshold: float = 1e-6,
    compile: Optional[Literal["xla"]] = None,
):
    kernel = model.kernel
    selection_func = make_inducing_selection_function(kernel, compile=compile)

    def _initialize_ips():
        x, _ = data
        noise = tf.convert_to_tensor(model.likelihood.variance)
        iv, _ = run_uniform_greedy_selection(
            selection_func, x, subset_size, numips, noise, threshold, rng=rng
        )
        model.inducing_variable.Z.assign(iv)

    return _initialize_ips


def initialize_ips(
    rng,
    data,
    model,
    numips: int,
    subset_size: int,
    threshold: float = 1e-6,
):
    x, _ = data
    kernel = model.kernel
    noise = model.likelihood.variance.numpy()
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
    elif compile_type == "tvm":
        from tvm_utils import extract_tf_graph_def
        from tvm_utils import run_graph_using_tvm

        func_jit = tf.function(fn)
        func_graph_def = extract_tf_graph_def(func_jit)
        target = "cuda"
        target_host = None
        ctx = None

        def fn_tvm_runnable():
            return run_graph_using_tvm(
                func_graph_def, [], target=target, target_host=target_host, ctx=ctx
            )

        return fn_tvm_runnable

    return fn


def update_optimizer_logs(res: OptimizeResult, store: Dict):
    new_values = {
        "status": [res.status],
        "nit": [res.nit],
        "nfev": [res.nfev],
        "objective": [res.fun],
    }

    if store == {}:
        store.update(new_values.items())

    store.update({key: value + new_values[key] for key, value in store.items()})


def create_monitor(
    holdout_interval: int,
    logdir: OsPath,
    funcs: Optional[Dict[str, Callable]] = None,
):
    funcs_dict: Dict[str, Callable] = {} if funcs is None else funcs
    monitor_logdir = Path(logdir, "tb")
    monitor = Monitor(monitor_logdir, holdout_interval=holdout_interval)

    for name, func in funcs_dict.items():
        monitor.add_callback(name, func)

    return monitor


def numpy_results(metric_fn: Callable):
    def _map(value):
        if isinstance(value, tf.Tensor):
            return value.numpy()
        return np.array(value)

    values = metric_fn()
    return tf.nest.map_structure(_map, values)


def create_optimizer_metrics_func(values):
    if not isinstance(values, dict):
        raise ValueError("Expected dictionary")

    def _optimizer_metrics(*args, **kwargs):
        select_keys = ["nit", "nfev", "objective"]
        if values == {}:
            return
        if any([key not in values or values[key] == [] for key in select_keys]):
            return

        return {key: values[key][-1] for key in select_keys}

    return _optimizer_metrics


def create_metrics_func(model, data, compile_flag: CompileType, prefix: Optional[str] = None):
    prefix = "" if prefix is None else f"{prefix}_"

    def _metrics():
        x, y = data
        loss = model.training_loss()
        predictions, variance = model.predict_f(x)
        err = y - predictions
        logden = model.likelihood.predict_log_density(predictions, variance, y)
        err_sq = tf.square(err)
        rmse = tf.sqrt(tf.reduce_mean(err_sq))
        nlpd = -tf.reduce_mean(logden)
        return {f"{prefix}rmse": rmse, f"{prefix}nlpd": nlpd, "loss": loss}

    compiled_function = compile_function(_metrics, compile_flag)

    def _numpy_metrics(*args, **kwargs):
        return numpy_results(compiled_function)

    return _numpy_metrics


def create_keep_parameters_func(model: gpflow.models.SGPR, only_trainable: bool = False):
    def _params(*args, **kwargs):
        params = {}
        for name, param in parameter_dict(model).items():
            if only_trainable and not param.trainable:
                continue
            name = name.lstrip(".")
            params[name] = param.numpy()
        return params

    return _params
