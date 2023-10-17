import matplotlib.pyplot as plt
from typing import Union
import numpy as np
import os
import io
from PIL import Image


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display(
    images: np.ndarray,
    n=10,
    merge=True,
    size=(20, 3),
    cmap: Union[str, None] = "gray_r",
    as_type="float32",
    save_to:Union[str, None]=None,
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    # if images.max() > 1.0:
    #     images = images / 255.0
    # elif images.min() < 0.0:
    #     images = (images + 1.0) / 2.0
    images = (images - images.min()) / (images.max() - images.min())

    if merge:
        fig, axs = plt.subplots(1, n, figsize=size, gridspec_kw={'wspace': 0})
        for i in range(n):
            axs[i].imshow(images[i].astype(as_type), cmap=cmap)
            axs[i].axis("off")

        if save_to:
            if os.path.isdir(save_to):
                raise ValueError("save_to must be a file path, not a directory.")
            fig.savefig(save_to, bbox_inches="tight", pad_inches=0)
            print(f"\nSaved to {save_to}")
        else:
            plt.show(fig)
    else:
        for i in range(n):
            plt.axis("off")
            plt.imshow(images[i].astype(as_type), cmap=cmap)
            if save_to:
                if os.path.isfile(save_to):
                    raise ValueError("save_to must be a directory, not a file path.")
                plt.savefig(f"../output/gan_{i}.png")
                plt.close()
            else:
                plt.show()

