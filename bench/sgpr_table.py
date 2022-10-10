from pathlib import Path
import pandas as pd
import glob
from bench_utils import read_h5_into_dict
import numpy as np
from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def main():
    base_path = Path(__file__).parent
    pattern_path = "output/sgpr/{dataset}/1GB/sgpr_*ips*/*/"
    datasets = {"houseelectric": pd.DataFrame(), "3droad": pd.DataFrame()}
    for (name, df) in datasets.items():
        wildcard_path = base_path / pattern_path.format(dataset=name)
        paths = glob.glob(str(wildcard_path))
        for p in paths:
            p_path = Path(p)
            info = read_h5_into_dict(p_path / "info.h5")
            events_paths = glob.glob(str(p_path / "tb/events.*"))
            assert len(events_paths) == 1
            events_path = events_paths[0]
            event_names = ["timers/total_time", "timers/inner_timer"]
            events = parse_tensorboard(events_path, event_names)
            total_time = events["timers/total_time"]["value"][1:]
            inner_time = events["timers/inner_timer"]["value"][1:]

            total_time_argmax = total_time.argmax()
            inner_time_argmax = inner_time.argmax()
            total_time = pd.concat(
                [total_time[:total_time_argmax], total_time[total_time_argmax + 1 :]]
            )

            train_time = (total_time.sum() - inner_time.sum()) / 60**2
            metrics = info["final_metric"]
            keys = ["seed", "dataset_name", "numips", "dim_size", "test_size", "train_size"]
            subset_info = {k: v for k, v in info.items() if k in keys}
            selected_info = {"train_time": train_time, **subset_info, **metrics}
            new_row = pd.DataFrame(selected_info)
            df = pd.concat([df, new_row])

        aggr_keys = {
            "train_time": ("mean", "std"),
            "test_nlpd": ("mean", "std"),
            "test_rmse": ("mean", "std"),
        }

        df_selected = df.groupby(["dataset_name", "numips"]).agg(aggr_keys)
        print(df_selected)


if __name__ == "__main__":
    main()
