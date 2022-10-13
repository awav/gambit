from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

import sys
import json
from pathlib import Path
from typing_extensions import Literal
import click
import numpy as np
import tensorflow as tf
import gpflow

cur_dir = str(Path(__file__).expanduser().absolute().parent)
sys.path.append(cur_dir)

from transformer import Transformer
from clitypes import LogdirPath
from bench_utils import BenchRunner, store_dict_as_h5
from bench_sgpr_utils import compile_function, CompileType

__default_gambit_logs = "./logs_transformer_default"
__gpu_devices = tf.config.get_visible_devices("GPU")
__gpu_dev = __gpu_devices[0] if __gpu_devices else None


if __gpu_dev is not None:
    click.echo(f">>> GPU device information: {__gpu_dev}")
    click.echo(">>> Set GPU memory growth")
    tf.config.experimental.set_memory_growth(__gpu_dev, True)

# New version after renaming
# TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-exp-transformer" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=10GB --xla_tensor_split_size=1GB --xla_enable_hlo_passes_only=tensor-splitter,algebraic-rewriter,dce,broadcast-simplifier,cholesky_expander,triangular_solve_expander,bitcast_dtypes_expander,CallInliner,gpu_scatter_expander,rce-optimizer" python ./exp_transformer.py --sequence-len 10000 2>&1 | tee output-exp-transformer.log

# New version after renaming
# TF_CPP_MIN_LOG_LEVEL=0 CUDA_VISIBLE_DEVICES="3" DUMPDIR="xla-exp-transformer" XLA_FLAGS="--xla_dump_hlo_as_dot --xla_dump_to=${DUMPDIR} --xla_tensor_size_threshold=20GB --xla_tensor_split_size=10GB" python ./exp_transformer.py --sequence-len 10000 2>&1 | tee output-exp-transformer.log


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
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit(image_size, data_augmentation, patch_size: int = 16):
    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 16
    num_epochs = 1
    patch_size = patch_size  # Size of the patches to be extract from the input images
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
    num_patches = (image_size // patch_size) ** 2

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


@click.command()
@click.option("-l", "--logdir", type=LogdirPath(), default=__default_gambit_logs)
@click.option("-m", "--memory-limit", type=str)
@click.option("-sl", "--sequence-len", type=int)
@click.option("-s", "--seed", type=int, default=0)
@click.option("-r", "--repeat", type=int, default=1)
@click.option("-w", "--warmup", type=int, default=1)
@click.option("-c", "--compile", default="xla", help="Compile function with xla, tf or none")
def main(
    sequence_len: int,
    memory_limit: int,
    logdir: str,
    seed: int,
    warmup: int,
    repeat: int,
    compile: Literal["xla", "tf", "none"],
):
    memory_limit = "none" if memory_limit is None else memory_limit
    info = {
        "sequence_len": sequence_len,
        "memory_limit": memory_limit,
        "seed": seed,
        "repeat": repeat,
        "warmup": warmup,
        "compile": compile,
    }
    info_str = json.dumps(info, indent=2)
    click.echo("===> Starting")
    click.echo(f"-> {info_str}")
    assert Path(logdir).exists()

    compile_flag: CompileType = compile if compile != "none" else None

    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    gpflow_dtype = gpflow.config.default_float()

    def ctt(x, dtype=None):
        dtype = gpflow_dtype if dtype is None else dtype
        return tf.convert_to_tensor(x, dtype=dtype)

    batch_size = 64
    d_model = 512
    num_heads = 8
    input_shape = (batch_size, sequence_len)

    # Max size is prod([batch_size, num_heads, seq_len, seq_len, d_model]) * precision

    # q = rng.randn(*input_shape)
    # k = rng.randn(*input_shape)
    # v = rng.randn(*input_shape)
    # q_tf = ctt(q)
    # k_tf = ctt(k)
    # v_tf = ctt(v)

    x = rng.uniform(0, 200, size=input_shape)
    y = rng.uniform(0, 200, size=input_shape)
    x_tf = ctt(x)
    y_tf = ctt(y)

    transformer = Transformer(
        num_layers=2,
        d_model=d_model,
        num_heads=num_heads,
        dff=2048,
        input_vocab_size=8500,
        target_vocab_size=8000,
    )

    def eval_test(inputs, targets):
        out = transformer([inputs, targets], training=False)
        return out

    eval_test_compiled = compile_function(eval_test, compile_flag)

    bench_runner = BenchRunner(repeat=repeat, warmup=warmup, logdir=logdir)
    results = bench_runner.bench(eval_test_compiled, [x_tf, y_tf])
    bench_table = {**info, **results}

    filepath = Path(logdir, "bench.h5")
    store_dict_as_h5(bench_table, filepath)

    if "elapsed_stats" not in results or "mem_stats" not in results:
        click.echo("⚠️ No stats in the benchmark output ⚠️ ")
        raise click.exceptions.Exit(0)

    (elap_mu, elap_std) = results["elapsed_stats"]
    (mem_mu, mem_std) = results["mem_stats"]

    click.echo(
        "[Bench] Total stat, "
        f"spent_avg={elap_mu}, spent_std={elap_std}, "
        f"mem_avg={mem_mu}, mem_std={mem_std}"
    )


if __name__ == "__main__":
    main()
