import tensorflow as tf


def outerprod(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.matmul(x, y, transpose_b=True)