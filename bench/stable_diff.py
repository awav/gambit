# %%
import os
import tensorflow as tf

def set_gpu(gpu_index: int):
    gpu_name = f"GPU:{gpu_index}"
    gpu_dev = [dev for dev in tf.config.get_visible_devices("GPU") if dev.name.endswith(gpu_name)]
    tf.config.set_visible_devices(gpu_dev, "GPU")

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
set_gpu(3)

# %%
import keras_cv
import matplotlib.pyplot as plt

# %%
def main(size: int, batch: int = 3):
    """
    Example from https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion
    """
    text = "High-resolution photo of a scientist solving the most difficult mathematical problem while drinking vodka in Hawaii"
    model = keras_cv.models.StableDiffusion(img_width=size, img_height=size, jit_compile=True)
    images = model.text_to_image(text, batch_size=batch)

    def plot_images(images):
        plt.figure(figsize=(20, 20))
        for i in range(len(images)):
            ax = plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i])
            plt.axis("off")

    plot_images(images)


batch = 3
size = 512
main(size, batch=batch)