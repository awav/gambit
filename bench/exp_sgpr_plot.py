from typing import Dict, List
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from bench_utils import select_from_report_data, expand_paths_with_wildcards


@click.group()
def main():
    pass


__metric_choices = click.Choice(["test_rmse", "test_nlpd"])


@main.command()
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def metric_vs_numips(files):
    click.echo("===> start")
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)
    join_by = ["dataset_name", "compile", "grad_ips"]
    select_keys = ["seed", "numips", "final_metric"]
    data = select_from_report_data(expanded_files, join_by, select_keys)

    seed_key = "seed"
    metric_key = "metric"

    def combine_by_ips(ips: List[int], seed: List[int], metrics: List[Dict]) -> Dict:
        combined_ips = {ip: {seed_key: [], metric_key: []} for ip in ips}

        for i, ip in enumerate(ips):
            combined_ips[ip][seed_key].append(seed[i])
            combined_ips[ip][metric_key].append(metrics[i])

        for ip in combined_ips:
            metrics: List = combined_ips[ip][metric_key]
            metrics = tf.nest.map_structure(lambda *vals: vals, *metrics)
            combined_ips[ip][metric_key] = metrics

        return combined_ips

    def summary_by_ips(
        combined_by_ips: Dict[int, Dict], sum_type: Literal["avg", "quantile"] = "avg"
    ):
        avg_metric = {}

        def summary(value):
            if sum_type == "avg":
                return (np.mean(value), np.std(value))
            elif sum_type == "quantile":
                return np.quantile(value, [0.25, 0.5, 0.75])
            raise NotImplementedError(f"Unknown summary type: {sum_type}")

        for ips_key, df in combined_by_ips.items():
            metric = {key: summary(values) for key, values in df[metric_key].items()}
            avg_metric[ips_key] = metric
        return avg_metric

    plt.rcParams.update(
        {
            "font.size": 9,
            # "figure.subplot.left": 0,
            # "figure.subplot.bottom": 0,
            "figure.subplot.right": 1,
            "figure.subplot.top": 1,
        }
    )
    figsize = (9, 3)
    fig_rmse, (ax_rmse, ax_nlpd) = plt.subplots(1, 2, figsize=figsize)
    # fig_nlpd, ax_nlpd = plt.subplots(1, 1, figsize=figsize)
    yticks = []
    xticks = []

    for key, fields in data.items():
        dataname, backend, grad_ips = key
        ips = np.array(fields["numips"])
        metric = np.array(fields["final_metric"])
        seed = np.array(fields["seed"])
        indices = np.argsort(ips)
        ips_sorted = np.take_along_axis(ips, indices, axis=0)
        seed_sorted = list(np.take_along_axis(seed, indices, axis=0))
        metric_sorted = list(np.take_along_axis(metric, indices, axis=0))

        combined = combine_by_ips(ips_sorted, seed_sorted, metric_sorted)
        avg = summary_by_ips(combined)
        ips_sorted_unique = np.unique(ips_sorted)

        def get_test_metric(name):
            metrics = [avg[ip][name] for ip in ips_sorted_unique]
            mu, std = zip(*metrics)
            mu = np.array(mu)
            std = np.array(std)
            return mu, std

        rmse_mu, rmse_std = get_test_metric("test_rmse")
        nlpd_mu, nlpd_std = get_test_metric("test_nlpd")

        yticks += list(rmse_mu)
        xticks += list(ips_sorted_unique)

        # rmse_min = np.min(rmse_mu)
        # rmse_max = np.max(rmse_mu)
        # tick_frac_size = 10
        # rmse_tick_frac = (rmse_max - rmse_min) / tick_frac_size
        # pows = np.linspace(0, 10, 20)
        # tick_steps = np.array([0, *[np.power(2, p) for p in pows]]) * rmse_tick_frac + rmse_min

        line = ax_rmse.plot(ips_sorted_unique, rmse_mu)
        color = line[0].get_color()

        scatter_settings = scatter_settings_from_key(key, color)

        ax_rmse.scatter(ips_sorted_unique, rmse_mu, **scatter_settings)
        ax_rmse.legend()

        ax_nlpd.plot(ips_sorted_unique, nlpd_mu)
        ax_nlpd.scatter(ips_sorted_unique, nlpd_mu, **scatter_settings)
        ax_nlpd.legend()

        print(f"-> key={key}")
        print(f"=> ips={ips_sorted}")
        print(f"=> seed={seed_sorted}")
        print(f"=> metric={metric_sorted}")
        print(f"=> combined={combined}")
        print(f"=> avg={avg}")
        print(f"=> test_rmse_mu={rmse_mu}")
        print(f"=> test_rmse_std={rmse_std}")

    xticks = np.sort(np.unique(xticks))
    # ax_rmse.set_xticks(xticks)
    # ax_nlpd.set_xticks(xticks)

    ax_rmse.xaxis.set_major_locator(tkr.MultipleLocator(2000))
    ax_rmse.xaxis.set_minor_locator(tkr.MultipleLocator(500))
    ax_rmse.yaxis.set_major_locator(tkr.MultipleLocator(0.05))
    ax_rmse.yaxis.set_minor_locator(tkr.MultipleLocator(0.01))
    ax_rmse.set_ylabel("RMSE")
    ax_rmse.set_xlabel("Number of inducing points")

    ax_nlpd.xaxis.set_major_locator(tkr.MultipleLocator(2000))
    ax_nlpd.xaxis.set_minor_locator(tkr.MultipleLocator(500))
    ax_nlpd.yaxis.set_major_locator(tkr.MultipleLocator(0.1))
    ax_nlpd.yaxis.set_minor_locator(tkr.MultipleLocator(0.05))
    ax_nlpd.set_ylabel("NLPD")
    ax_nlpd.set_xlabel("Number of inducing points")

    border = 800
    minx, maxx = -500, border
    shade_color = "tab:red"
    shade_alpha = 0.1

    ax_rmse.axvspan(minx, maxx, alpha=shade_alpha, color=shade_color)
    ax_rmse.axvline(border, alpha=shade_alpha + 0.3, color=shade_color, linestyle="--")
    ax_rmse.set_xlim(0, np.max(xticks) + 500)
    ax_rmse.yaxis.grid(visible=True, which="both", linestyle=":")
    ax_rmse.spines['right'].set_visible(False)
    ax_rmse.spines['top'].set_visible(False)

    ax_nlpd.axvspan(minx, maxx, alpha=shade_alpha, color=shade_color)
    ax_nlpd.axvline(border, alpha=shade_alpha + 0.3, color=shade_color, linestyle="--")
    ax_nlpd.set_xlim(0, np.max(xticks) + 500)
    ax_nlpd.yaxis.grid(visible=True, which="both", linestyle=":")
    ax_nlpd.spines['right'].set_visible(False)
    ax_nlpd.spines['top'].set_visible(False)

    # ax_rmse.grid(axis="y")
    # ax_nlpd.grid(axis="y")
    plt.tight_layout(w_pad=0.5)
    plt.show()
    click.echo("<== finished")


def label_from_key(key) -> str:
    _, backend, grad_ips = key
    label_suffix = " (trainable ips)" if grad_ips else ""
    label = f"{backend_name(backend)}{label_suffix}"
    return label


def scatter_settings_from_key(key, color) -> Dict:
    label = label_from_key(key)
    marker_key = key[1:]
    settings = dict(cmap=color, label=label)
    markers = {
        ("tf", True): dict(marker="o", facecolors="none", edgecolors=color),
        ("tf", False): dict(marker="o", facecolors="none", edgecolors=color),
        ("xla", True): dict(marker="."),
        ("xla", False): dict(marker="."),
    }
    return {**settings, **markers[marker_key]}


def backend_name(backend: str) -> str:
    names = {"xla": "TF-eXLA", "tf": "TF"}
    return names[backend]


if __name__ == "__main__":
    main()
