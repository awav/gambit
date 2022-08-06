from operator import itemgetter
import click
import numpy as np
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
    sequence_len = "sequence_len"
    select_keys = [mem_stats_key, elapsed_stats_key, sequence_len]
    data = select_from_report_data(expanded_files, join_by, select_keys)

    new_data = {}
    for ((mem_key, _), values) in data.items():
        mems = values[mem_stats_key]
        sizes = np.array(values[sequence_len], dtype=int)
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
        new_data[mem_key][sequence_len] = sizes_sorted

    none_key = "none"
    xla_none_key = "xla-none"
    other_key = "tl10GB_ts1GB"
    new_data = {
        "XLA": new_data[xla_none_key],
        "eXLA": new_data[other_key],
        "TF": new_data[none_key],
    }

    tf_key = "TF"
    xla_key = "XLA"
    exla_key = "eXLA"

    linestyles = {
        tf_key: ":",
        xla_key: "-",
        exla_key: (0, (5, 5)),
    }

    colors = {
        tf_key: "tab:green",
        xla_key: "tab:orange",
        exla_key: "tab:blue",
    }

    plt.rcParams.update(
        {
            "font.size": 9,
            # "figure.subplot.left": 0,
            # "figure.subplot.bottom": 0,
            "figure.subplot.right": 1,
            "figure.subplot.top": 1,
        }
    )

    # figsize = (4.3, 4.5)
    # fig, (ax_mem, ax_time) = plt.subplots(2, 1, sharex=True, figsize=figsize)
    figsize = (8, 2.7)
    fig, (ax_mem, ax_time) = plt.subplots(1, 2, figsize=figsize)

    def plot_values(ax1, ax2, data, key):
        values = data[key]
        mem = values[mem_stats_key][:, 0]
        time_values = values[elapsed_stats_key]
        time_mean = time_values[:, 0]
        time_std = time_values[:, 1]
        sequences = values[sequence_len]
        linestyle = linestyles[key]
        color = colors[key]
        ax1.plot(sequences, mem, color=color, linestyle=linestyle, label=key)
        ax2.plot(sequences, time_mean, color=color, linestyle=linestyle, label=key)
        return sequences

    plot_values(ax_mem, ax_time, new_data, "XLA")
    plot_values(ax_mem, ax_time, new_data, "TF")
    sequences = plot_values(ax_mem, ax_time, new_data, "eXLA")

    ax_mem.set_ylabel("Memory, bytes")
    ax_time.set_ylabel("Elapsed time, seconds")
    ax_mem.set_xlabel("Sequence length")
    ax_time.set_xlabel("Sequence length")

    ax_mem.yaxis.set_major_locator(tkr.MultipleLocator(5e9))
    ax_mem.yaxis.set_minor_locator(tkr.MultipleLocator(1e9))

    # ax_mem.legend(loc="center right")
    ax_time.legend(loc="upper left")
    ax_mem.yaxis.grid(visible=True, which="both", linestyle=":")
    ax_time.yaxis.grid(visible=True, which="both", linestyle=":")

    ax_mem.set_xticks(sequences, labels=sequences, rotation=90)
    ax_time.set_xticks(sequences, labels=sequences, rotation=90)

    ax_mem.spines["right"].set_visible(False)
    ax_mem.spines["top"].set_visible(False)
    ax_time.spines["right"].set_visible(False)
    ax_time.spines["top"].set_visible(False)

    stop_color = "black"
    ax_time.axvline(2000, alpha=0.5, color=stop_color)
    ax_mem.axvline(2000, alpha=0.5, color=stop_color)

    fig.suptitle("Transformer model, https://www.tensorflow.org/text/tutorials/transformer")

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
