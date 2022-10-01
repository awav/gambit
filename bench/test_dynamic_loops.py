import tensorflow as tf


def main():
    n = 100
    m = 133
    k = 15
    A = tf.random.normal([n, m])
    v = tf.random.normal([m, 1])
    steps = m // k

    def cond(i, _):
        return i <= steps
    
    def body(i, acc):
        """
        Args:
            i: Current iteration of the loop.
            acc: Tensor accumulator.
        
        Returns:
            Tuple of a new iteration and an updated accumulator.
        """
        start = i * k
        end = (i + 1) * k
        As = A[:, start:end]
        vs = v[start:end]
        new_part = As @ vs
        new_acc = acc + new_part
        return [i + 1, new_acc]

    def loop(A):
        n = tf.shape(A)[0]
        i = tf.cast(0, tf.int32)
        o = tf.zeros([n, 1], dtype=A.dtype)
        loop_vars = [i, o]
        res = tf.while_loop(cond, body, loop_vars)
        return res[0], res[1]

    tf_loop = tf.function(loop, jit_compile=True)
    # tf_loop = tf.function(loop)

    i0, res0 = loop(A)
    i1, res1 = tf_loop(A)
    print("done")


if __name__ == "__main__":
    main()