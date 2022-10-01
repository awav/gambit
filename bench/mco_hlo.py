# TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="0" DUMPDIR="xla-mco" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_enable_hlo_passes_only=mco,dce,broadcast-simplifier,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander,CallInliner,gpu_scatter_expander,rce-optimizer" python ./mco_hlo.py 2>&1 | tee mco-hlo.logs


import tensorflow as tf


@tf.function(jit_compile=True)
def mco(a, b, c):
    return tf.reduce_sum(a @ b @ c, axis=-1)


def main():
    n = 100
    m = 75
    p = 50
    k = 25
    a = tf.random.normal((n, m))
    b = tf.random.normal((m, p))
    c = tf.random.normal((p, k))
    result = mco(a, b, c)
    result.numpy()


if __name__ == "__main__":
    main()
