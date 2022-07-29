import tensorflow as tf
import numpy as np

Tensor = tf.Tensor


# TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-attention" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_try_split_tensor_size=3MB --xla_enable_hlo_passes_only=split-intermediate-tensors,algebraic-rewriter,dce,broadcast-simplifier,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander,CallInliner,gpu_scatter_expander,rce-optimizer" python ./attention.py 2>&1 | tee output-attention.log


def self_attention(queries: Tensor, keys: Tensor, values: Tensor) -> Tensor:
    """
    Args:
        queries: vector R^{m \times d}
        keys: matrix R^{n \times d}
        values: matrix R^{n \times d}
    Return:
        set of vectors R^{m \times d}
    """
    s = tf.matmul(queries, keys, transpose_b=True)  # R^{m \times n}
    s_sm = tf.math.softmax(s, axis=-1)  # R^{m \times n}, s_i = \exp(s_i) / (\sum_j^n \exp(s_j))
    result = s_sm @ values
    return result


if __name__ == "__main__":
    # flag = "test"
    flag = "attention"

    if flag == "test":
        d = 3
        m = 9
        n = 10
        query = tf.random.uniform([1, d])
        keys = tf.random.uniform([n, d])
        values = tf.random.uniform([n, d])
        queries = tf.convert_to_tensor([query, query])

        att = self_attention(query, keys, values).numpy()
        self_att = self_attention(queries, keys, values).numpy()
        att = np.squeeze(att)
        self_att_0 = np.squeeze(self_att[0])
        self_att_1 = np.squeeze(self_att[1])
        np.testing.assert_almost_equal(att, self_att_0)
        np.testing.assert_almost_equal(att, self_att_1)
    elif flag == "attention":
        self_attention_jit = tf.function(self_attention, jit_compile=True)
        # try this one if it does not work:
        # self_attention_jit = tf.function(self_attention, experimental_compile=True)
        d = 10
        m = 1000
        n = 1000
        queries = tf.random.uniform([m, d])
        keys = tf.random.uniform([n, d])
        values = tf.random.uniform([n, d])

        result = self_attention_jit(queries, keys, values)
        result_numpy = result.numpy()
    
    print("finished")
