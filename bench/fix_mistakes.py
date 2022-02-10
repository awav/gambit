import click
import shutil
from pprint import pprint
from bench_utils import expand_paths_with_wildcards, read_h5_into_dict, store_dict_as_h5
from copy import deepcopy


@click.command()
@click.argument("files", nargs=-1, type=click.Path(dir_okay=False))
def fix_mistakes(files):
    files = files if isinstance(files, (list, tuple)) else [files]
    expanded_files = expand_paths_with_wildcards(files)
    keys_to_fix = {"test__nlpd": "test_nlpd", "test__rmse": "test_rmse"}
    final_metric = "final_metric"
    for h5file in expanded_files:
        change = False
        data = read_h5_into_dict(h5file)
        for broken_key, correct_key in keys_to_fix.items():

            if broken_key in data[final_metric]:
                change = True
                value = data[final_metric].pop(broken_key)
                data[final_metric][correct_key] = value

        if change:
            shutil.copyfile(h5file, f"{h5file}.bkp")
            store_dict_as_h5(data, h5file)


if __name__ == "__main__":
    fix_mistakes()