from typing import Callable, Sequence, Union, Dict, Tuple
from pathlib import Path
import tensorflow as tf
import gpflow
from tensorboardX import SummaryWriter
import numpy as np
from torch.functional import Tensor


class Monitor:
    def __init__(
        self,
        logdir: Union[Path, str],
        holdout_interval: int = 1,
        init_iteration: int = 0,
        max_queue: int = 5,
        flush_secs: int = 10,
    ):
        self._logdir = logdir
        self._iter: int = init_iteration
        self._writer = SummaryWriter(logdir=str(logdir), max_queue=max_queue, flush_secs=flush_secs)
        self._callbacks = {}
        self._holdout_interval: int = holdout_interval

    @property
    def writer(self) -> SummaryWriter:
        return self._writer
    
    @property
    def callbacks(self) -> Dict[str, Tuple[Callable, Dict]]:
        return self._callbacks

    def add_callback(self, name: str, callback: Callable):
        self._callbacks[name] = (callback, {})

    def reset(self):
        self._iter = 0
        self.writer.flush()
        callbacks = {}
        for name, (cb, _) in self._callbacks.items():
            callbacks[name] = (cb, {})
        self._callbacks = callbacks

    def flush(self):
        self.writer.flush()

    def close(self):
        self.flush()
        self.writer.close()

    def _incr_iteration(self):
        self._iter += 1

    def collect_logs(self) -> Dict:
        out = {}
        for cb_name, (_, logs) in self._callbacks.items():
            if isinstance(logs, dict):
                out[cb_name] = logs
        return out

    def _handle_callback(self, step: int, name: str):
        cb, logs = self._callbacks[name]
        results = cb(step)
        for key, value in results.items():
            idx = f"{name}/{key}"
            if isinstance(value, (tf.Tensor, gpflow.Parameter)):
                numpy_value = value.numpy()
            else:
                numpy_value = np.array(value)

            if isinstance(numpy_value, (list, np.ndarray)) and _len(numpy_value) > 1:
                for i, r in enumerate(value):
                    self.writer.add_scalar(f"{idx}_{i}", r, global_step=step)
            else:
                self.writer.add_scalar(idx, numpy_value, global_step=step)

            if key in logs:
                logs[key].append(numpy_value)
            else:
                logs[key] = [numpy_value]

    def __call__(self, step: int, *args, **kwargs):
        cur_step = self._iter
        interval = self._holdout_interval
        if interval > 0 and cur_step % interval == 0:
            for name in self._callbacks:
                self._handle_callback(cur_step, name)
        self._incr_iteration()


def _len(obj) -> int:
    if isinstance(obj, np.ndarray):
        return obj.size
    return len(obj)


def store_logs(path: Path, logs: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, logs, allow_pickle=True)
