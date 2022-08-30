import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import sys
import json
from pathlib import Path
from typing_extensions import Literal
import click
import numpy as np
import gpflow

from clitypes import LogdirPath
from bench_utils import BenchRunner, store_dict_as_h5
from bench_sgpr_utils import (
    compile_function,
    CompileType
)

__default_gambit_logs = "./logs_vit_default"

__gpu_devices = tf.config.get_visible_devices("GPU")
__gpu_dev = __gpu_devices[0] if __gpu_devices else None

if __gpu_dev is not None:
    tf.config.experimental.set_memory_growth(__gpu_dev, True)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches



class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-m", "--memory-limit", type=str)
@click.option("-s", "--seed", type=int, default=0)
@click.option("-r", "--repeat", type=int, default=1)
@click.option("-w", "--warmup", type=int, default=1)
@click.option("-c", "--compile", default="none", help="Compile function with xla, tf or none")
@click.option("-d", "--dump-name",  default="v2")
@click.option("-t", "--bench-type",  default="batch")
def main(
    memory_limit: str,
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    compile: Literal["xla", "tf", "none"],
    dump_name: Literal["v2", "v1", "none"],
    bench_type: Literal["batch", "epoch"],
):
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 16
    num_epochs = 1
    patch_size = 16  # Size of the patches to be extract from the input images
    projection_dim = 32
    num_heads =12
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 12
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
    num_classes = 100
    input_shape = (32, 32, 3)
    def create_vit_classifier():
        inputs = layers.Input(shape=input_shape)
        # Augment data.
        augmented = data_augmentation(inputs)
        # Create patches.
        patches = Patches(patch_size)(augmented)
        # Encode patches.
        encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

        # Create multiple layers of the Transformer block.
        for _ in range(transformer_layers):
            # Layer normalization 1.
            x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
            # Create a multi-head attention layer.
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            # Skip connection 1.
            x2 = layers.Add()([attention_output, encoded_patches])
            # Layer normalization 2.
            x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
            # MLP.
            x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
            # Skip connection 2.
            encoded_patches = layers.Add()([x3, x2])

        # Create a [batch_size, projection_dim] tensor.
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        # Classify outputs.
        logits = layers.Dense(num_classes)(features)
        # Create the Keras model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model


    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

    print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
    if bench_type == "batch":
        dump_file_path = __default_gambit_logs+"/vit_stat_"+dump_name+".txt"
    else:
        dump_file_path = __default_gambit_logs+"/vit_stat_epoch_"+dump_name+".txt"
    mco_stat = []
    if dump_name != "none":
        image_size_list = [32,64,96,128,160,192,224,256,288,320,352,384,400]
    else:
        image_size_list = [32,64,96,128,160,192,224,256,288]
    for image_size in image_size_list:
        num_patches = (image_size // patch_size) ** 2
        memory_limit = "none" if memory_limit is None else memory_limit
        info = {
            "batch_size":batch_size,
            "image_size":image_size,
            "patch_size":patch_size,
            "memory_limit": memory_limit,
            "projection_dim":projection_dim,
            "num_heads":num_heads,
            "seed": seed,
            "repeat": repeat,
            "warmup": warmup,
            "compile": compile,
            "bench_type":bench_type
        }
        info_str = json.dumps(info, indent=2)
        click.echo("===> Starting")
        click.echo(f"-> {info_str}")
        assert Path(logdir).exists()

        compile_flag: CompileType = compile if compile != "none" else None

        data_augmentation = keras.Sequential(
            [
                layers.Normalization(),
                layers.Resizing(image_size, image_size),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(factor=0.02),
                layers.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
        # Compute the mean and the variance of the training data for normalization.
        data_augmentation.layers[0].adapt(x_train)

        vit_classifier = create_vit_classifier()
        optimizer = tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            )
        vit_classifier.compile(
                optimizer=optimizer,
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[
                    keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                    keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
                ],
            )
        if compile_flag!= None:
            tf.config.optimizer.set_jit(True)
        bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
        if bench_type == "batch":
            results = bench_runner.bench(vit_classifier.train_on_batch, [x_train[:batch_size], y_train[:batch_size],None,None,True,False])
        else:
            results = bench_runner.bench(vit_classifier.fit, [x_train,y_train,batch_size,num_epochs,'auto',None,0.1])
        bench_table = {**info, **results}

        filepath = Path(logdir, "bench.h5")
        store_dict_as_h5(bench_table, filepath)

        if "elapsed_stats" not in results or "mem_stats" not in results:
            click.echo("⚠️ No stats in the benchmark output ⚠️ ")
            raise click.exceptions.Exit(0)

        (elap_mu, elap_std) = results["elapsed_stats"]
        (mem_mu, mem_std) = results["mem_stats"]
        if __gpu_dev is not None:
            # turn into Mib
            mem_mu, mem_std = mem_mu/1024/1024, mem_std/1024/1024

        mco_stat.append([image_size,elap_mu*1000,mem_mu])
        click.echo(
            "[Bench] Total stat, "
            f"spent_avg={elap_mu}, spent_std={elap_std}, "
            f"mem_avg={mem_mu}, mem_std={mem_std}"
        )
        np.savetxt(dump_file_path, mco_stat, fmt='%f', delimiter=',')
    
    
if __name__ == "__main__":
    main()

