import tensorflow as tf

def genvector(n):
    return tf.ones((n, 1), dtype=tf.float64)

def split_outerproduct_full(x, y, v, k: int = 10):
    n = x.shape[0]
    x = x + genvector(n)
    y = y + genvector(n)
    v = v + genvector(n)

    interm = tf.matmul(y, x, transpose_b=True)
    return interm @ v


def split_outerproduct(x, y, v, k: int = 10):
    n = x.shape[0]

    xi = tf.split(x, k)
    vi = tf.split(v, k)

    def kprod(y, x, i, outerprod=True):
        return tf.matmul(y, x[i], transpose_b=outerprod)

    with tf.name_scope("MapPhase"):
        with tf.device("/device:gpu:1"):
            interm0 = tf.exp(kprod(y, xi, 0))
            interm1 = tf.exp(kprod(y, xi, 1))
            interm2 = tf.exp(kprod(y, xi, 2))
            interm3 = tf.exp(kprod(y, xi, 3))
            interm4 = tf.exp(kprod(y, xi, 4))
        with tf.device('/device:gpu:2'):
            interm5 = tf.exp(kprod(y, xi, 5))
            interm6 = tf.exp(kprod(y, xi, 6))
            interm7 = tf.exp(kprod(y, xi, 7))
            interm8 = tf.exp(kprod(y, xi, 8))
            interm9 = tf.exp(kprod(y, xi, 9))


    with tf.name_scope("ReducePhase1"):
        with tf.device('/device:gpu:1'):
            out0 = kprod(interm0, vi, 0, outerprod=False)
            out1 = kprod(interm1, vi, 1, outerprod=False)
            out2 = kprod(interm2, vi, 2, outerprod=False)
            out3 = kprod(interm3, vi, 3, outerprod=False)
            out4 = kprod(interm4, vi, 4, outerprod=False)
        with tf.device('/device:gpu:2'):
            out5 = kprod(interm5, vi, 5, outerprod=False)
            out6 = kprod(interm6, vi, 6, outerprod=False)
            out7 = kprod(interm7, vi, 7, outerprod=False)
            out8 = kprod(interm8, vi, 8, outerprod=False)
            out9 = kprod(interm9, vi, 9, outerprod=False)

    with tf.name_scope("ReducePhase2"):
        return (out0 + out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8 + out9)


def main(n: int, full: bool = False):
    x = tf.Variable(genvector(n), dtype=tf.float64)
    y = tf.Variable(genvector(n), dtype=tf.float64)
    v = tf.Variable(genvector(n), dtype=tf.float64)

    # @tf.function(experimental_compile=True)
    @tf.function()
    def run():
        if full:
            vec = split_outerproduct_full(x, y, v)
        else:
            vec = split_outerproduct(x, y, v)

        return tf.reduce_sum(vec)

    # Warm up
    run()
    folder = "logdir"
    with tf.profiler.experimental.Profile(folder):
        return [run().numpy() for _ in range(5)]


if __name__ == "__main__":
    main(20000)
