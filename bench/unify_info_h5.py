import click
from bench_utils import read_h5_into_dict, store_dict_as_h5, expand_paths_with_wildcards


@click.command()
@click.option("--grad/--no-grad", default=False)
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def main(files, grad: bool):
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)
    for filepath in expanded_files:
        data = read_h5_into_dict(filepath)
        backup_filepath = f"{filepath}.backup"
        store_dict_as_h5(data, backup_filepath)
        metric = data["final_metric"]
        metric_copy = metric.copy()
        for key in metric:
            if key.startswith("test__"):
                value = metric_copy[key]
                new_key = key.replace("test__", "test_", 1)
                metric_copy.pop(key)
                metric_copy[new_key] = value
        data["final_metric"] = metric_copy
        data["grad_ips"] = grad
        store_dict_as_h5(data, filepath)


if __name__ == "__main__":
    main()
