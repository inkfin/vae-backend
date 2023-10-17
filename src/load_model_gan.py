import torch
import numpy as np
import matplotlib.pyplot as plt
from gan import Generator
from utils import display

# Set random seed for reproducibility
seed = 6969
# seed = random.randint(1, 10000) # use if you want new results
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_list_pick = [0, 1, 2, 4, 5, 6, 7, 9, 12, 14, 19, 22, 38, 41, 42, 43, 44, 46, 52, 59]

def load_model():
    load_netG: Generator = torch.load("../models/netG_100.pth").to(device)
    fixed_noise: torch.Tensor = torch.load("../models/fixed_noise.pt").to(device)
    print("fixed_noise loaded:", fixed_noise.shape)
    fixed_noise_pick = fixed_noise[img_list_pick]
    return load_netG, fixed_noise_pick

def interp_noise(load_netG, n, noise1, noise2):
    interp_array = []
    for i in range(n):
        interp_array.append(noise1 * (n - i) / (n - 1) + noise2 * i / (n - 1))

    interp_array = torch.stack(interp_array)

    with torch.no_grad():
        fake_list = load_netG(interp_array).detach().cpu()

    fixed_fake_array: np.ndarray = np.transpose(fake_list.numpy(), (0, 2, 3, 1))
    fixed_fake_array = (fixed_fake_array - fixed_fake_array.min()) / (
        fixed_fake_array.max() - fixed_fake_array.min()
    )

    display(fixed_fake_array, save_to="../output/interp_gan.png")
    # plt.figure(figsize=(15, 6))
    # for i in range(n):
    #     plt.subplot(2, 5, i + 1)
    #     plt.axis("off")
    #     plt.imshow(fixed_fake_array[i])
    # plt.show()

def interp_gan(load_netG, fixed_noises, n, idx1, idx2):
    noise1 = fixed_noises[idx1]
    noise2 = fixed_noises[idx2]

    interp_array = []
    for i in range(n):
        interp_array.append(noise1 * (n - i) / (n - 1) + noise2 * i / (n - 1))

    interp_array = torch.stack(interp_array)

    with torch.no_grad():
        fake_list = load_netG(interp_array).detach().cpu()

    fixed_fake_array: np.ndarray = np.transpose(fake_list.numpy(), (0, 2, 3, 1))

    display(fixed_fake_array, save_to="../output/interp_gan.png")

    fixed_fake_array = (fixed_fake_array - fixed_fake_array.min()) / (
        fixed_fake_array.max() - fixed_fake_array.min()
    )
    return fixed_fake_array

if __name__ == "__main__":
    load_netG, fixed_noises = load_model()

    interp_gan(load_netG, fixed_noises, 10, 4, 8)

