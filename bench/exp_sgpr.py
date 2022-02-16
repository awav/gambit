import sys
import os
import json
import pprint
from pathlib import Path
from typing_extensions import Literal
import click
from gpflow.utilities import set_trainable
import numpy as np
import tensorflow as tf

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import LogdirPath
from bench_utils import get_uci_dataset, store_dict_as_h5, tf_data_tuple
from barelybiasedgp.scipy_copy import Scipy

from bench_sgpr_utils import (
    make_initialize_ips_function,
    initialize_sgpr,
    create_metrics_func,
    create_keep_parameters_func,
    create_optimizer_metrics_func,
    create_monitor,
    update_optimizer_logs,
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


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-s", "--seed", type=int, default=0)
@click.option("-d", "--dataset-name", type=DatasetChoices, default="keggundirected")
@click.option("-np", "--num_phases", type=int, default=1)
@click.option("-ni", "--maxiter_per_phase", type=int, default=100)
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
@click.option("-ss", "--subset-size", default=100000, help="Greedy selection subset size")
@click.option("-mhi", "--metric-holdout-interval", type=int, default=50)
@click.option("-hi", "--holdout-interval", default=1, type=int)
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
    metric_holdout_interval: int,
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
        "num_phases": num_phases,
        "maxiter_per_phase": maxiter_per_phase,
        "numips": numips,
        "grad_ips": False,
        "compile": compile,
        "subset_size": subset_size,
        "metric_holdout_interval": metric_holdout_interval,
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

    opt_logs = {}
    param_logs = {}
    metric_logs = {}

    monitor_optimizer_fn = create_optimizer_metrics_func(opt_logs)
    monitor_all_param_fn = create_keep_parameters_func(model, only_trainable=False)
    monitor_metric_fn = create_metrics_func(model, tf_data_test, compile_flag, prefix="test")

    monitor = create_monitor(holdout_interval, logdir)
    monitor.add_callback("param", monitor_all_param_fn)

    if metric_holdout_interval >= 1:
        monitor.add_callback("metric", monitor_metric_fn, metric_holdout_interval)

    rng = np.random.RandomState(seed)
    set_trainable(model.inducing_variable, False)

    train_vars = model.trainable_variables
    scipy_options = dict(maxiter=maxiter_per_phase)
    loss_fn = model.training_loss_closure(compile=False)

    threshold = 1e-6
    initialize_ips_fn = make_initialize_ips_function(
        rng, data, model, numips, subset_size, threshold=threshold
    )

    click.echo("==> Start phases")
    for ph in range(num_phases):
        click.echo(f"---> Phase {ph}")
        rng = np.random.RandomState(seed)
        initialize_ips_fn()

        click.echo(f"---> optimize for phase {ph}")
        opt = Scipy()
        res = opt.minimize(
            loss_fn,
            train_vars,
            step_callback=monitor,
            compile=compile_flag,
            options=scipy_options,
        )

        update_optimizer_logs(res, opt_logs)
        monitor.handle_callback("optimizer", monitor_optimizer_fn)
        monitor.handle_callback("metric", monitor_metric_fn, metric_logs)
        monitor.handle_callback("param", monitor_all_param_fn, param_logs)

        if res.nit < 2:
            click.echo(
                f"---> Break training loop after phase {ph}."
                f"The number of optimization iterations is at minimum: {res.nit}"
            )
            break

        click.echo("-> optimisation result:")
        click.echo(res)

    click.echo("---> Record after optimization")
    final_metric_logs = {}
    monitor.handle_callback("metric", monitor_metric_fn, final_metric_logs)

    # optimization traces
    opt_path = Path(logdir, "opt.h5")
    store_dict_as_h5(opt_logs, opt_path)

    # trainable parameters
    param_path = Path(logdir, "param.h5")
    store_dict_as_h5(param_logs, param_path)
    info_extension = {f"param_path": str(param_path)}

    metric_path = Path(logdir, "metric.h5")
    store_dict_as_h5(metric_logs, metric_path)
    info_extension = {f"metric_path": str(metric_path), **info_extension}

    # monitor metrics
    if metric_holdout_interval >= 1:
        metric_path = Path(logdir, "metric_full.h5")
        metric_full_log = monitor.collect_logs()["metric"]
        store_dict_as_h5(metric_full_log, metric_path)
        info_extension = {f"metric_full_path": str(metric_path), **info_extension}

    info_extension = {"final_metric": final_metric_logs, **info_extension}
    info_extended = {**info, **info_extension}
    info_path = Path(logdir, "info.h5")
    store_dict_as_h5(info_extended, info_path)

    click.echo("<=== Finished")


if __name__ == "__main__":
    pprint.pprint(dict(os.environ), width=1)
    main()
