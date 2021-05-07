import tensorflow as tf

def genvector(n):
    return tf.ones((n, 1), dtype=tf.float64)

def main(n: int):
    u = tf.Variable(0.1, dtype=tf.float64)
    x = tf.Variable(genvector(n), dtype=tf.float64)
    y = tf.Variable(genvector(n), dtype=tf.float64)

    @tf.function(experimental_compile=True)
    def run():
        z = x @ tf.transpose(y)
        v0 = 0.1 * tf.concat([genvector(n), genvector(n)], axis=1)
        v1 = u * tf.eye(n, dtype=tf.float64)
        s = tf.transpose(tf.linalg.triangular_solve(z, v0, lower=True)) @ v1
        return tf.reduce_sum(s)

    # Warm up
    run()

    folder = "logdir"
    with tf.profiler.experimental.Profile(folder):
        return [run().numpy() for _ in range(5)]


if __name__ == "__main__":
    main(1000)