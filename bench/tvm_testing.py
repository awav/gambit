from typing import Optional
import tvm
import tensorflow as tf
import numpy as np
from tvm_utils import extract_tf_graph_def, run_graph_using_tvm


def main():
    def test_func(a: tf.Tensor, b: tf.Tensor):
        ab = tf.matmul(a, b, transpose_b=True)
        return tf.reduce_sum(tf.exp(0.1 * ab + 0.1))

    tf.random.set_seed(111)
    n = 1000003
    m = 100
    a_np = np.random.randn(n, m)
    b_np = np.random.randn(n, m)
    a = tf.convert_to_tensor(a_np, dtype=tf.float32)
    b = tf.convert_to_tensor(b_np, dtype=tf.float32)

    signature = [
        tf.TensorSpec(shape=(n, m), dtype=tf.float32),
        tf.TensorSpec(shape=(n, m), dtype=tf.float32),
    ]
    func_jit = tf.function(test_func, input_signature=signature)
    func_graph_def = extract_tf_graph_def(func_jit, a, b)

    target = "cuda"
    target_host = None
    ctx = tvm.cuda(0)
    # ctx = None

    try:
        print("----------------> TVM")
        result = run_graph_using_tvm(
            func_graph_def, [a, b], target=target, target_host=target_host, ctx=ctx
        )
        print(f"TVM result: {result}")
    except Exception as error:
        print(">===============>")
        print(str(error))
        print("<===============<")

    print("----------------> XLA")
    func_xla = tf.function(test_func, jit_compile=True)
    result = func_xla(a, b)
    print(f"XLA result: {result.numpy()}")
    print("==> finished")


if __name__ == "__main__":
    main()
