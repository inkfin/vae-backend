import matplotlib.pyplot as plt
from typing import Union


def sample_batch(dataset):
    batch = dataset.take(1).get_single_element()
    if isinstance(batch, tuple):
        batch = batch[0]
    return batch.numpy()


def display(
    images, n=10, size=(20, 3), cmap:Union[str,None] ="gray_r", as_type="float32", save_to=None
):
    """
    Displays n random images from each one of the supplied arrays.
    """
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    fig, axs = plt.subplots(1, n, figsize=size)
    for i in range(n):
        axs[i].imshow(images[i].astype(as_type), cmap=cmap)
        axs[i].axis("off")

    if save_to:
        fig.savefig(save_to)
        print(f"\nSaved to {save_to}")
    else:
        plt.show(fig)
