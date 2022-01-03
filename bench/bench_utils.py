from functools import reduce
from operator import iconcat
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
import h5py
import re
import glob

import tensorflow.config.experimental as tf_exp
from dataclasses import dataclass
from time import time
from pathlib import Path
from memory_profiler import memory_usage

FilePath = Union[Path, str]


@dataclass
class BenchRunner:
    repeat: int
    warmup: int
    logdir: FilePath

    def bench(self, func_to_run: Callable, prepare_args: Union[Callable, Sequence]):
        gpu_devices = tf.config.get_visible_devices("GPU")
        if len(gpu_devices) > 1:
            raise RuntimeError("Expected one GPU")
        
        if not callable(prepare_args):
            prepare_args_fn = lambda: prepare_args
        else:
            prepare_args_fn = prepare_args


        gpu_dev = gpu_devices[0] if gpu_devices else None
        if gpu_dev is not None:
            dev_type = "gpu"
            dev_name = gpu_dev.name.split(":", 1)[-1]

            def run_and_collect_stat():
                args = prepare_args_fn()
                time0 = time()
                _ = func_to_run(*args)
                elapsed = time() - time0
                mem = tf_exp.get_memory_info(dev_name)["peak"]
                return elapsed, mem

        else:
            dev_type = "cpu"
            dev_name = dev_type

            def run_and_collect_stat():
                args = prepare_args_fn()
                func_tuple = (func_to_run, args, {})
                time0 = time()
                mem_info = memory_usage(func_tuple)
                elapsed = time() - time0
                mem = np.max(mem_info)
                return elapsed, mem

        for _ in range(self.warmup):
            args = prepare_args_fn()
            func_to_run(*args)

        elaps, mems = [], []
        for _ in range(self.repeat):
            elapsed, mem = run_and_collect_stat()
            elaps.append(elapsed)
            mems.append(mem)

        elaps = np.array(elaps)
        mems = np.array(mems)

        output = dict(
            device=dev_type,
            elapsed_stats=(np.mean(elaps), np.std(elaps)),
            mem_stats=(np.mean(mems), np.std(mems)),
            elapsed_array=elaps,
            mem_array=mems,
        )

        return output


def parse_name(name: str):
    splits = name.split("_")[1:]
    re_compiled = re.compile(r"^([a-zA-Z]+)(\d+)$")

    def process(elem):
        match = re_compiled.match(elem)
        if not match:
            raise ValueError(f"Cannot parse element: {elem}")
        key = match.group(1)
        value = match.group(2)
        return key, value

    return dict([process(s) for s in splits])


def store_dict_as_h5(data: Dict, filepath: FilePath):
    with h5py.File(filepath, mode="w") as writer:
        for k, v in data.items():
            if isinstance(v, str):
                v = np.array(v, dtype="S")
            elif isinstance(v, int):
                v = np.array(v, dtype=int)
            else:
                v = np.array(v)
            writer.create_dataset(k, data=v)


def read_h5_into_dict(filepath: FilePath) -> Dict:
    data = {}
    with h5py.File(filepath) as reader:
        for k in reader.keys():
            data[k] = read_hdf_value(reader, k)
    return data


def read_h5(filepath: FilePath) -> h5py.File:
    return h5py.File(filepath)


def read_hdf_value(hdf: h5py.File, key: str):
    def _convert(value: Any):
        if value.dtype.kind == "S":
            return str(np.array(value, dtype="U"))
        elif value.dtype.kind == "i" and value.shape == ():
            return int(np.array(value))
        elif value.dtype.kind == "f" and value.shape == ():
            return float(np.array(value))
        return np.array(value)

    value = hdf.get(key)
    result = _convert(value)
    return result


def select_from_report_data(
    filepaths: Sequence[FilePath],
    join_by: Union[Sequence[str], Dict],
    select_field: str,
) -> Dict:
    store_at = dict()
    for filepath in filepaths:
        data = read_h5_into_dict(filepath)
        if not all([key in data for key in join_by]):
            raise ValueError(f"Some keys do not exist in HDF files: {join_by}")
        stop = False
        for key in join_by:
            if isinstance(join_by, dict):
                value = join_by[key]
                if value != Ellipsis and data[key] != value:
                    stop = True
                    break
        if stop:
            continue
        tuple_key = tuple([data[key] for key in join_by])
        if tuple_key not in store_at:
            store_at[tuple_key] = [data[select_field]]
        else:
            store_at[tuple_key].append(data[select_field])
    return store_at


def expand_paths_with_wildcards(filepaths: Sequence[str]) -> Sequence[str]:
    full_list = [glob.glob(f) for f in filepaths]
    return list(reduce(iconcat, full_list, []))
