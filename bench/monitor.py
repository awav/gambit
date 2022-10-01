from typing import Callable, Sequence, Union, Dict, Tuple, Optional
from pathlib import Path
import tensorflow as tf
import gpflow
from tensorboardX import SummaryWriter
import numpy as np
import math
from codetiming import Timer, TimerError


class MonitorTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self._total_timer = Timer("total_timer", logger=None)
        self._inner_timer = Timer("inner_timer", logger=None)
        self._logs = dict()

    def take_readings(self) -> Dict:
        time_metrics = dict()
        time_metrics["inner_timer"] = self.inner_timer.last
        try:
            self.total_timer.stop()
            time_metrics["total_time"] = self.total_timer.last
            self.total_timer.start()
        except TimerError:
            time_metrics["total_time"] = self.inner_timer.last
            self.total_timer.start()
        return time_metrics
    
    @property
    def logs(self) -> Dict:
        return self._logs

    @property
    def total_timer(self) -> Timer:
        return self._total_timer

    @property
    def inner_timer(self) -> Timer:
        return self._inner_timer


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
        self._callbacks: Dict[str, Tuple[Callable, Dict]] = {}
        self._callback_holdout_interval: Dict[str, int] = {}
        self._global_holdout_interval: int = holdout_interval
        self._monitor_timers = MonitorTimer()

    @property
    def timers(self) -> MonitorTimer:
        return self._monitor_timers

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    @property
    def callbacks(self) -> Dict[str, Tuple[Callable, Dict]]:
        return self._callbacks
    
    def start_timers(self):
        self.timers.total_timer.start()
    
    def timers_logs(self) -> Dict:
        return self.timers.logs

    def add_callback(self, name: str, callback: Callable, holdout_interval: Optional[int] = None):
        self._callbacks[name] = (callback, {})
        if holdout_interval is not None:
            self._callback_holdout_interval[name] = holdout_interval

    def reset(self):
        self._iter = 0
        self.writer.flush()
        callbacks = {}
        for name, (cb, _) in self._callbacks.items():
            callbacks[name] = (cb, {})
        self._callbacks = callbacks
        self._monitor_timers.reset()

    def flush(self):
        self.writer.flush()

    def close(self):
        self.flush()
        self.writer.close()

    def _incr_iteration(self):
        self._iter += 1

    def collect_logs(self) -> Dict:
        out = dict()
        out["timers"] = self.timers_logs()
        for cb_name, (_, logs) in self._callbacks.items():
            if isinstance(logs, dict):
                out[cb_name] = logs
        return out

    def handle_callback(self, name: str, callback: Callable, logs: Optional[Dict] = None):
        cur_step = self._iter
        self._handle_callback(cur_step, name, callback, logs)

    def handle_callbacks(self, callbacks: Dict[str, Union[Callable, Tuple[Callable, Dict]]]):
        cur_step = self._iter
        for name, value in callbacks.items():
            cb, logs = (value, None) if callable(value) else value
            self._handle_callback(cur_step, name, cb, logs)

    def _handle_callback(
        self,
        step: int,
        name: str,
        callback: Callable,
        logs: Optional[Dict] = None,
    ):
        results = callback(step)
        if results is None:
            return

        for key, value in results.items():
            idx = f"{name}/{key}"
            if isinstance(value, (tf.Tensor, gpflow.Parameter)):
                numpy_value = value.numpy()
            else:
                numpy_value = np.array(value)

            if isinstance(numpy_value, (list, np.ndarray)) and _len(numpy_value) > 1:
                if len(numpy_value.shape) == 1:
                    for i, r in enumerate(value):
                        self.writer.add_scalar(f"{idx}_{i}", r, global_step=step)
                else:
                    self.writer.add_histogram(idx, numpy_value, global_step=step)
            else:
                self.writer.add_scalar(idx, numpy_value, global_step=step)

            if logs is not None:
                if key in logs:
                    logs[key].append(numpy_value)
                else:
                    logs[key] = [numpy_value]

    def __call__(self, step: int, *args, **kwargs):
        with self.timers.inner_timer:
            cur_step = self._iter
            global_interval = self._global_holdout_interval

            for name, value in self._callbacks.items():
                cb_interval = None
                if name in self._callback_holdout_interval:
                    cb_interval = self._callback_holdout_interval[name]
                if cb_interval is not None:
                    if cur_step % cb_interval != 0:
                        continue
                elif cur_step % global_interval != 0:
                    continue

                cb, logs = value
                self.handle_callback(name, cb, logs)

        self.handle_callback("timers", lambda _: self.timers.take_readings(), self.timers.logs)
        self._incr_iteration()


def _len(obj) -> int:
    if isinstance(obj, np.ndarray):
        return obj.size
    return len(obj)


def store_logs(path: Path, logs: Dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, logs, allow_pickle=True)
