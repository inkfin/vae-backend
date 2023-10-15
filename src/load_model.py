import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import utils
from keras.models import load_model

from scipy.stats import norm
import pandas as pd

import imageio
from skimage.transform import resize

from vae import (
    VAE,
    IMAGE_SIZE,
    Z_DIM,
    NUM_FEATURES,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    BETA,
    LOAD_MODEL,
    encoder,
    decoder,
)

from utils import sample_batch, display
from vae_utils import get_vector_from_label, add_vector_to_images, morph_faces

# Ignore the warning:
import warnings

warnings.filterwarnings("ignore")


dataset_root = "../data/images"


# Preprocess the data
def preprocess(img):
    img = tf.divide(tf.cast(img, "float32"), 255.0)
    return img


def load_img(data_root: str) -> np.ndarray:
    # Load the data
    img_data = utils.image_dataset_from_directory(
        data_root,
        labels=None,
        color_mode="rgb",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        interpolation="bilinear",
    )
    img_prep = img_data.map(lambda x: preprocess(x))
    return img_prep


def load_vae() -> VAE:
    # Load your trained VAE model
    vae = load_model("../models/vae")
    return vae

def interp_images(vae: VAE, n, img1, img2):
    # interpolate by morphing latent spaces
    _, _, example_latent = vae.encoder.predict(np.array([img1, img2]))

    interp_latent = []
    for t in range(n):
        img = example_latent[0] * (1 - t / n) + example_latent[1] * (t / n)
        interp_latent.append(img)
    interp_latent = np.array(interp_latent)

    rec = np.array(vae.decoder(interp_latent))
    display(rec, n, save_to="../output/interp.png")



# main function
if __name__ == "__main__":
    img_prep = load_img(dataset_root)
    vae = load_vae()

    # img_sample = sample_batch(img_prep)  # Show some faces from the training set
    # display(img_sample, cmap=None, save_to="../output/real_faces.png")

    """
    latent_space = vae.encoder.predict(img_prep)
    latent_space = np.array(latent_space)

    # Assuming a 2D latent space
    plt.figure(figsize=(10, 8))
    plt.scatter(latent_space[:, 0], latent_space[:, 1], c="b", marker="o")
    plt.xlabel("Latent Space Dimension 1")
    plt.ylabel("Latent Space Dimension 2")
    plt.title("Latent Space Visualization")
    plt.savefig("../output/latent_space.png")
    plt.close()
    """

    # Select a subset of the test set
    batches_to_predict = 1
    example_images = np.array(
        list(img_prep.take(batches_to_predict).get_single_element())
    )

    # # Create autoencoder predictions and display
    # z_mean, z_log_var, reconstructions = vae.predict(example_images)
    # display(example_images, save_to="../output/Example real faces.png")
    # display(reconstructions, save_to="../output/Example reconstructions.png")

    image1 = imageio.imread('../data/images/1_2000.jpg')
    image2 = imageio.imread('../data/images/17_2000.jpg')
    
    image1 = resize(image1, (IMAGE_SIZE, IMAGE_SIZE))
    image2 = resize(image2, (IMAGE_SIZE, IMAGE_SIZE))

    print(image1.shape)
    print(image2.shape)

    # interp_images(vae, 10, example_images[0], example_images[1])
    interp_images(vae, 10, image1, image2)