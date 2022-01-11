from typing import Dict, List
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from bench_utils import select_from_report_data, expand_paths_with_wildcards


@click.group()
def main():
    pass


__metric_choices = click.Choice(["test_rmse", "test_nlpd"])


@main.command()
# @click.option("-m", "--metric", type=__metric_choices, default="test_rmse")
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def metric_vs_numips(files):
    click.echo("===> start")
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)
    join_by = ["dataset_name", "compile"]
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

    fig, ax = plt.subplots(1, 1)

    for key, fields in data.items():
        dataname, backend = key
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
        test_rmse = [avg[ip]["test_rmse"] for ip in ips_sorted_unique]
        test_rmse_mu, test_rmse_std = zip(*test_rmse)
        test_rmse_mu = np.array(test_rmse_mu)
        test_rmse_std = np.array(test_rmse_std)

        line = ax.plot(ips_sorted_unique, test_rmse_mu)
        color = line[0].get_color()

        for i, ip in enumerate(ips_sorted_unique):
            mu = test_rmse_mu[i]
            std = test_rmse_std[i]
            mu_min, mu_max = (mu - std), (mu + std)
            s = 10
            ip_min = ip - s
            ip_max = ip + s
            ax.vlines(ip, mu_min, mu_max, color=color)
            ax.hlines(mu_min, ip_min, ip_max)
            ax.hlines(mu_max, ip_min, ip_max)

        print(f"-> key={key}")
        print(f"=> ips={ips_sorted}")
        print(f"=> seed={seed_sorted}")
        print(f"=> metric={metric_sorted}")
        print(f"=> combined={combined}")
        print(f"=> avg={avg}")
        print(f"=> test_rmse_mu={test_rmse_mu}")
        print(f"=> test_rmse_std={test_rmse_std}")

    plt.tight_layout()
    plt.show()
    click.echo("<== finished")


if __name__ == "__main__":
    main()
