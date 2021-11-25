import gpflow
import tensorflow as tf
from wrapt.wrappers import wrap_object_attribute
from .kernels import create_kernel

import numpy as np
from gpflow.config import default_float
import bayesian_benchmarks.data as bbd


def get_dataset(name: str):
    dat = getattr(bbd, name)(prop=0.67)
    train, test = (dat.X_train, dat.Y_train), (dat.X_test, dat.Y_test)
    x_train, y_train = _norm_dataset(train)
    x_test, y_test = _norm_dataset(test)
    return (_to_float(x_train), _to_float(y_train)), (_to_float(x_test), _to_float(y_test))


def create_sgpr_loss_and_grad(dataset_name: str, kernel_name: str, num_ip: int, dtype=None, do_grad: bool = True):
    train_data, test_data = get_dataset(dataset_name)
    x, _ = train_data
    dim = x.shape[-1]
    num = x.shape[0]

    print(f"@@@@@@ Dataset={dataset_name}, size={num}, grad={do_grad}")

    kernel = create_kernel(kernel_name, dim, dtype)
    n = train_data[0].shape[0]
    m_indices = np.random.choice(n, size=num_ip, replace=False)
    ip = train_data[0][m_indices]
    train_data_tf = (_to_tf(train_data[0]), _to_tf(train_data[1]))
    model = gpflow.models.SGPR(train_data_tf, kernel, inducing_variable=ip)

    loss_fn = model.training_loss_closure(compile=False)
    variables = model.trainable_variables

    def loss_and_grad():
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(variables)
            loss = loss_fn()
        grads = tape.gradient(loss, variables)
        return [loss, *grads]

    if do_grad:
        exec_func = loss_and_grad
    else:
        exec_func = loss_fn

    return exec_func


def create_training_sgpr_procedure():
    pass


def _to_tf(arr: np.ndarray):
    return tf.convert_to_tensor(arr)


def to_tf_data(data):
    return _to_tf(data[0]), _to_tf(data[1])


def _to_float(arr: np.ndarray):
    return arr.astype(default_float())


def _norm(x: np.ndarray) -> np.ndarray:
    """Normalise array with mean and variance."""
    mu = np.mean(x, axis=0, keepdims=True)
    std = np.std(x, axis=0, keepdims=True) + 1e-6
    return (x - mu) / std


def _norm_dataset(data):
    """Normalise dataset tuple."""
    return _norm(data[0]), _norm(data[1])