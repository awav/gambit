from pathlib import Path
import numpy as np
import click

__default_gambit_logs = "./default_gambit_logs"


class FloatType(click.ParamType):
    name = "float_type"

    def convert(self, value, param, ctx):
        options = {"fp32": np.float32, "fp64": np.float64}
        try:
            norm_value = value.lower()
            float_type = options[norm_value]
            return float_type
        except Exception as ex:
            self.fail(f"{value} is not a valid float type [fp32, fp64]", param, ctx)

    def __repr__(self):
        return "FloatType"


class MemoryLimit(click.ParamType):
    name = "memory_limit"

    def convert(self, value, param, ctx):
        if value is None:
            return value

        options = {"mb": 1, "gb": 1024}
        suffixes = tuple(options.keys())
        try:
            if value.lower().endswith(suffixes):
                return int(value)
            return int(value)
        except:
            self.fail(f"{value} is not a valid float type (allowed fp32, and fp64)", param, ctx)

    def __repr__(self):
        return "MemoryLimit"


class Shape(click.ParamType):
    name = "shape"

    def convert(self, value, param, ctx):
        try:
            values = value.lstrip(" (").rstrip(") ").split(",")
            values = [int(float(v)) for v in values]
            return tuple(values)
        except ValueError:
            self.fail(f"{value} is not in valid shape format", param, ctx)

    def __repr__(self):
        return "FloatType"


class LogdirPath(click.Path):
    def __init__(self, mkdir: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.mkdir = mkdir

    def convert(self, value, param, ctx):
        logdir = super().convert(value, param, ctx)
        logdir_path = Path(logdir).expanduser().resolve()
        if self.mkdir:
            logdir_path.mkdir(parents=True, exist_ok=True)
        return logdir_path
