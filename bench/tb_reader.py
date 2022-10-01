from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


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
    path = "/home/artem/code/gambit/bench/output/sgpr/3droad/1GB/sgpr_trainable_ips10000_i1000/111/tb/events.out.tfevents.1664472858.piocbggpu02"
    names = ["timers/total_time", "timers/inner_timer"]
    values = parse_tensorboard(path, names)
    print()


if __name__ == "__main__":
    main()