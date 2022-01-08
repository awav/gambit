from functools import reduce
from operator import iconcat
from typing import Any, Callable, Dict, Sequence, Tuple, Union
import numpy as np
import tensorflow as tf
from gpflow.config import default_float
import h5py
import re
import glob

import tensorflow.config.experimental as tf_exp
from dataclasses import dataclass
from time import time
from pathlib import Path
from memory_profiler import memory_usage
import bayesian_benchmarks.data as bbd


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
    splits = name.split("_")
    header = splits[0]
    rest = splits[1:]
    re_compiled = re.compile(r"^([a-zA-Z]+)(\d+)$")

    def process(elem):
        match = re_compiled.match(elem)
        if not match:
            raise ValueError(f"Cannot parse element: {elem}")
        key = match.group(1)
        value = match.group(2)
        return key, value

    return {"name": header, **dict([process(s) for s in rest])}


def store_dict_as_h5(data: Dict, filepath: FilePath):
    with h5py.File(filepath, mode="w") as writer:
        for root_key, root_value in data.items():
            store_hdf_value(writer, root_key, root_value)


def read_h5_into_dict(filepath: FilePath) -> Dict:
    data = {}
    with h5py.File(filepath) as reader:
        for key in reader.keys():
            read_hdf_value(data, reader, key)
    return data


def read_h5(filepath: FilePath) -> h5py.File:
    return h5py.File(filepath)


HdfStruct = Union[h5py.File, h5py.Group]


def store_hdf_value(writer: HdfStruct, key: str, value: Any):
    if isinstance(value, dict):
        for sub_key, sub_value in value.items():
            if not isinstance(sub_key, str):
                raise NotImplementedError(f"String keys allowed only. {sub_key} passed")
            store_hdf_value(writer, f"{key}/{sub_key}", sub_value)
        return

    if isinstance(value, str):
        value = np.array(value, dtype="S")
    elif isinstance(value, int):
        value = np.array(value, dtype=int)
    else:
        value = np.array(value)

    writer.create_dataset(key, data=value)


def read_hdf_value(out_dict: Dict[str, Any], hdf: HdfStruct, key: str):
    def _convert(value: Any):
        if isinstance(value, h5py.Group):
            sub_out_dict = {}
            for sub_key in value.keys():
                read_hdf_value(sub_out_dict, value, sub_key)
            return sub_out_dict

        if value.dtype.kind == "S":
            return str(np.array(value, dtype="U"))
        elif value.dtype.kind == "i" and value.shape == ():
            return int(np.array(value))
        elif value.dtype.kind == "f" and value.shape == ():
            return float(np.array(value))
        return np.array(value)

    value = hdf.get(key)
    result = _convert(value)
    out_dict[key] = result


def select_from_report_data(
    filepaths: Sequence[FilePath],
    join_by: Union[Sequence[str], Dict],
    select_fields: Sequence[str],
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
            fields_init = {field: [data[field]] for field in select_fields}
            store_at[tuple_key] = fields_init
        else:
            fields = store_at[tuple_key]
            fields_update = {field: values + [data[field]] for field, values in fields}
            store_at[tuple_key] = fields_update
    return store_at


def expand_paths_with_wildcards(filepaths: Sequence[str]) -> Sequence[str]:
    full_list = [glob.glob(f) for f in filepaths]
    return list(reduce(iconcat, full_list, []))


def get_uci_dataset(name: str, seed: int):
    full_name = f"Wilson_{name}"
    dat = getattr(bbd, full_name)(prop=0.67)
    train, test = (dat.X_train, dat.Y_train), (dat.X_test, dat.Y_test)
    x_train, y_train = _norm_dataset(train)
    x_test, y_test = _norm_dataset(test)
    train = _to_gpflow_default_float(x_train), _to_gpflow_default_float(y_train)
    test = _to_gpflow_default_float(x_test), _to_gpflow_default_float(y_test)
    return train, test


def _norm(x: np.ndarray) -> np.ndarray:
    """Normalise array with mean and variance."""
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-9
    return (x - mu) / std


def _norm_dataset(data):
    """Normalise dataset tuple."""
    return _norm(data[0]), _norm(data[1])


def _to_gpflow_default_float(arr: np.ndarray):
    return arr.astype(default_float())


def to_tf_scope(arr: np.ndarray):
    return tf.convert_to_tensor(arr)


def tf_data_tuple(data):
    return to_tf_scope(data[0]), to_tf_scope(data[1])
