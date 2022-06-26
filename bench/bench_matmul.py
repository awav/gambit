import sys
from pathlib import Path
from typing import Dict, Tuple, NamedTuple
import click
import numpy as np
import tensorflow as tf
import gpflow

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from clitypes import FloatType, LogdirPath

Dataset = Tuple[np.ndarray, np.ndarray]
DatasetBundle = NamedTuple


_gpu_devices = tf.config.get_visible_devices("GPU")
_gpu_dev = _gpu_devices[0] if _gpu_devices else None

# if _gpu_dev is not None:
#     print(f">>> GPU device information: {_gpu_dev}")
#     print(">>> Set GPU memory growth")
#     tf.config.experimental.set_memory_growth(_gpu_dev, True)

__default_gambit_logs = "./default_gambit_logs"


# XLA_FLAGS="--xla_tensor_size_threshold=100MB --xla_enable_hlo_passes_only=tensor-splitter,broadcast-simplifier,dce,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander --xla_dump_hlo_as_text --xla_dump_hlo_as_dot --xla_dump_to=./xla-test-100m" python bench_matmul.py -s 999 -n 1000753 -d 1000

# XLA_FLAGS="--xla_tensor_size_threshold=100MB" python bench_matmul.py -s 999 -n 1000753 -d 1000
# XLA_FLAGS="--xla_tensor_size_threshold=10GB" python bench_matmul.py -s 999 -n 1000753 -d 1000

# XLA_FLAGS="--xla_tensor_size_threshold=100MB" python bench_matmul.py -s 999 -n 1000753 -d 1000
# XLA_FLAGS="--xla_tensor_size_threshold=10GB" python bench_matmul.py -s 999 -n 1000753 -d 1000

@click.command()
@click.option("-s", "--seed", type=int, default=None)
@click.option("-n", "--num", type=int, help="The size of input vectors")
@click.option("-d", "--dim", type=int, help="The dimention of input vectors")
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("--jit/--no-jit", default=True, help="Compile function with or without XLA")
def main(
    num: int,
    dim: int,
    logdir: str,
    seed: int,
    jit: bool,
):
    assert Path(logdir).exists()
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    f = np.random.rand(dim, dim)
    l_factor = np.linalg.cholesky(f.T @ f + np.eye(dim))
    lhs = np.random.randn(dim, num)
    b_res = l_factor @ lhs
    m = np.random.rand(dim, num)
    b = b_res + m

    a = tf.Variable(np.random.randn(num, 1), dtype=tf.float64)
    b = tf.Variable(b, dtype=tf.float64)
    m = tf.Variable(m, dtype=tf.float64)
    c = tf.Variable(np.random.randn(1, dim), dtype=tf.float64)
    l = tf.Variable(l_factor, dtype=tf.float64)

    def test_fn():
        add_a_c = a + c
        t = b - m
        s = tf.linalg.triangular_solve(l, t)
        d = tf.linalg.matmul(s, add_a_c)
        return tf.reduce_mean(d)

    fn_to_test = tf.function(test_fn, jit_compile=jit)
    res = fn_to_test()
    value = res.numpy()

    print(f"Test value: {value}")


if __name__ == "__main__":
    main()
