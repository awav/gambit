from typing import Sequence
import click
import pandas as pd
import numpy as np
from bench_utils import select_from_report_data, expand_paths_with_wildcards


@click.group()
def main():
    pass


@main.command()
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def churn_speed_table(files):
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)
    cols = ["dataset_name", "distance", "dataset_size", "dim_size"]
    join_by = ["dataset", "query_size", "backend", *cols]
    data = select_from_report_data(expanded_files, join_by, ["elapsed_stats"])

    backend_id = join_by.index("backend")
    backends = list(set([tuple_key[backend_id] for tuple_key, _ in data.items()]))
    backends_size = len(backends)
    col_names = ["Dataset", "Distance", "n", "d"]
    col_names += backends

    rows = {}

    for tuple_key, selected_values in data.items():
        values = selected_values["elapsed_stats"]
        i = backend_id + 1
        _, query_size, backend = tuple_key[:i]
        row_key = tuple_key[i:]
        backend_value = np.mean([v for (v, _) in values])
        backend_value = query_size / backend_value
        k = backends.index(backend)
        if row_key not in rows:
            rows[row_key] = [None] * backends_size
        rows[row_key][k] = backend_value
    
    array = [[*keys, *values] for keys, values in rows.items()]
    df = pd.DataFrame(array, columns=col_names)

    click.echo(f"{df.to_latex()}")


if __name__ == "__main__":
    main()