import click
from bench_utils import select_from_report_data, expand_paths_with_wildcards


@click.group()
def main():
    pass


__metric_choices = click.Choice(["test_rmse", "test_nlpd"])


@main.command()
@click.option("-m", "--metric", type=__metric_choices, default="test_rmse")
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def metric_vs_numips(files):
    click.echo("===> start")
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)
    join_by = ["dataset_name", "compile"]
    select_keys = ["seed", "numips", "metric"]
    data = select_from_report_data(expanded_files, join_by, select_keys)
    click.echo(data)
    click.echo("<== finished")


if __name__ == "__main__":
    main()
