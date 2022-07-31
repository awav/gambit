import tensorflow as tf

def main():
    # @tf.function(jit_compile=True)
    def reshape_dot_reshape(x, y):
        # x_ = tf.reshape(x, [-1, x.shape[-1]])
        # y_ = tf.reshape(y, [y.shape[0], -1])
        x_ = x
        y_ = y
        z = tf.matmul(x_, y_)
        return tf.reshape(z, [*x.shape[:3], -1])

    x = tf.random.normal((2, 3, 1, 1, 2000, 1000))
    y = tf.random.normal((1, 1, 4, 5, 1000, 1))

    z = reshape_dot_reshape(x, y).numpy()
    print("Finished")


if __name__ == "__main__":
    main()