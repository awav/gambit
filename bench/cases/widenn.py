from typing import Tuple
import tensorflow as tf


def create_widenn(num_units: int, output_size: int, input_shape: Tuple):
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_units, activation="relu"),
        tf.keras.layers.Dense(output_size),
    ])
    return model

def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 256
    x_test = x_test.astype('float32') / 256

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    return ((x_train, y_train), (x_test, y_test))

(x_train, y_train), (x_test, y_test) = load_data()


tf.config.optimizer.set_jit("autoclustering")
model = create_widenn(1000, 10, x_train.shape[1:])


if __name__ == "__main__":
    print(f"model.output_shape={model.output_shape}")