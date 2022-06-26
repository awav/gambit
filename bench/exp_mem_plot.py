from operator import itemgetter
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

    new_data = {}
    for ((mem_key, _), values) in data.items():
        mems = values[mem_stats_key]
        sizes = np.array(values[size_key], dtype=int)
        elapses = values[elapsed_stats_key]
        mems_new = np.stack(mems)
        elapses_new = np.stack(elapses)

        ids = np.argsort(sizes)
        sizes_sorted = np.take_along_axis(sizes, ids, axis=0)
        mems_sorted = mems_new[ids, ...]
        elapses_sorted = elapses_new[ids, ...]

        new_data[mem_key] = {}
        new_data[mem_key][mem_stats_key] = mems_sorted
        new_data[mem_key][elapsed_stats_key] = elapses_sorted
        new_data[mem_key][size_key] = sizes_sorted

    plt.rcParams.update(
        {
            "font.size": 9,
            # "figure.subplot.left": 0,
            # "figure.subplot.bottom": 0,
            "figure.subplot.right": 1,
            "figure.subplot.top": 1,
        }
    )

    keys = list(new_data.keys())

    cmap = "tab20c"
    n = len(keys)
    colors = plt.get_cmap(cmap)(np.arange(n, dtype=int))
    marker_types = ["o", "^", "x", "v"]
    m = len(marker_types)
    k = n // m + int(bool(n % m))
    markers = (marker_types * k)[:n]

    mem_keys = sort_memory_sizes(keys)
    # figsize = (4.3, 4.5)
    # fig, (ax_mem, ax_time) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    figsize = (8, 2.7)
    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=figsize)
    for i, mem_key in enumerate(mem_keys):
        values = new_data[mem_key]
        mem = values[mem_stats_key][:, 0]
        elapsed_time = values[elapsed_stats_key][:, 0]
        sizes = values[size_key]
        color = colors[i]
        marker = markers[i]
        ax_mem.plot(sizes, mem, color=color, marker=marker, markersize=5, label=mem_key)
        ax_time.plot(sizes, elapsed_time, color=color, marker=marker, markersize=5, label=mem_key)

    ax_mem.set_ylabel("Memory, bytes")
    ax_time.set_ylabel("Elapsed time, seconds")
    ax_mem.set_xlabel("Size, $n$")
    ax_time.set_xlabel("Size, $n$")

    ax_mem.yaxis.set_major_locator(tkr.MultipleLocator(5e9))
    ax_mem.yaxis.set_minor_locator(tkr.MultipleLocator(1e9))

    # ax_mem.legend(loc="center right")
    ax_time.legend(loc="upper left")
    ax_mem.yaxis.grid(visible=True, which="both", linestyle=":")
    ax_time.yaxis.grid(visible=True, which="both", linestyle=":")

    ax_mem.xaxis.set_major_locator(tkr.MultipleLocator(1e5))
    ax_mem.xaxis.set_minor_locator(tkr.MultipleLocator(5e4))
    ax_time.xaxis.set_major_locator(tkr.MultipleLocator(1e5))
    ax_time.xaxis.set_minor_locator(tkr.MultipleLocator(5e4))

    ax_mem.spines["right"].set_visible(False)
    ax_mem.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.show()

    click.echo("<== finished")


def parse_mem(size: str) -> int:
    scales = {"GB": 1e9, "MB": 1e6, "KB": 1e3}
    for m, scale in scales.items():
        if size.endswith(m):
            return int(size.strip(m)) * scale
    raise RuntimeError(f"Value {size} could not be parsed")


def sort_memory_sizes(mem_list):
    indices = list(range(len(mem_list)))
    key_func = lambda i: parse_mem(mem_list[i])
    indices.sort(key=key_func, reverse=True)
    sorted_fn = itemgetter(*indices)
    return sorted_fn(mem_list)


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
