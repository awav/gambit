from typing import Dict, List
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from bench_utils import select_from_report_data, expand_paths_with_wildcards


@click.command()
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def main(files):
    click.echo(f"===> {__file__}")
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)
    join_by = ["memory_limit", "seed"]
    mem_stats_key = "mem_stats"
    elapsed_stats_key = "elapsed_stats"
    size_key = "size"
    select_keys = [mem_stats_key, elapsed_stats_key, size_key]
    data = select_from_report_data(expanded_files, join_by, select_keys)

    for (mem_key, values) in data.items():
        mems = values[mem_stats_key]
        elapses = values[elapsed_stats_key]
        mems_new = np.stack(mems)
        elapses_new = np.stack(elapses)

        data[mem_key][mem_stats_key] = mems_new
        data[mem_key][elapsed_stats_key] = elapses_new

    fig, ax = plt.subplots(1, 1)
    for ((mem_key, seed), values) in data.items():
        mem = values[mem_key][:, 0]
        sizes = values[size_key]
        ax.plot(mem, sizes, label=mem_key)
    
    ax.legend()
    plt.show()

    click.echo(f"{data}")
    click.echo(f"{list(data.keys())}")
    click.echo("<== finished")


def parse_mem(size: str) -> int:
    scale = ["GB", "MB", "KB"]
    pass


def label_from_key(key) -> str:
    _, backend, grad_ips = key
    label_suffix = " (trainable ips)" if grad_ips else ""
    label = f"{backend_name(backend)}{label_suffix}"
    return label


def marker_from_key(key) -> str:
    marker_key = key[1:]
    markers = {
        ("tf", True): "X",
        ("tf", False): "X",
        ("xla", True): ".",
        ("xla", False): ".",
    }
    return markers[marker_key]


def backend_name(backend: str) -> str:
    names = {"xla": "TF-eXLA", "tf": "TF"}
    return names[backend]


if __name__ == "__main__":
    main()
