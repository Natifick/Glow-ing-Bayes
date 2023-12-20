import torch
import torch.nn
import numpy as np

import torchvision.utils as TVutils

import os
import imageio


def save_gif(images: torch.Tensor, path: str):
    images = images.cpu().detach().numpy()
    images = [image.transpose(1, 2, 0) for image in images]
    imageio.mimsave(path, images, fps=10)


@torch.no_grad()
def get_latent_direction(model, dataset, attr_list, n_samples=300):
    """
    Obtain direction from {no attr} to {attr} in latent space
    """
    model.eval()
    cum_mean_from = [0]*4
    cum_mean_to = [0]*4

    n_from, n_to = 0, 0

    neg_indices = np.random.choice(np.where(np.array(attr_list) == -1)[0], n_samples, replace=False)
    pos_indices = np.random.choice(np.where(np.array(attr_list) == 1)[0], n_samples, replace=False)

    for i in neg_indices:
        img = dataset[i].cuda()[None,:]
        z = model.forward(img)[2]
        for idx in range(len(z)):
            cum_mean_from[idx] = (z[idx] + cum_mean_from[idx]*n_from)/(n_from + 1)
        n_from += 1
    
    for i in pos_indices:
        img = dataset[i].cuda()[None,:]
        z = model.forward(img)[2]
        for idx in range(len(z)):
            cum_mean_to[idx] = (z[idx] + cum_mean_to[idx]*n_to)/(n_to + 1)
        n_to += 1
    
    return [(c_to - c_from)/torch.norm(c_to - c_from) for c_from, c_to in zip(cum_mean_from, cum_mean_to)]


@torch.no_grad()
def sample_direction(model, img, direction, steps, step_size, filename=None, save_path='./sample_directions/', render_gif=False):
    """Sample in direction in latent space."""
    model.eval()
   
    # obtain latent
    z = model.forward(img)[2]

    # sample in direction
    samples = []
    for n in range(1, steps + 1):
        z_shifted = []
        for i in range(len(z)):
            z_shifted.append(z[i] + n*step_size*direction[i])
        img_shifted = model.reverse(z_shifted)
        samples.append(img_shifted.cpu().detach())
    samples = torch.cat(samples, dim=0)

    if filename is not None:
        os.makedirs('./sample_directions', exist_ok=True)
        if render_gif:
            save_gif(samples, save_path + filename + '.gif')
        else:
            TVutils.save_image(samples, save_path + filename +'.png', normalize=True, nrow=steps)


@torch.no_grad()
def sample_linear_interpolation(model, img1, img2, steps, filename=None, save_path='./sample_intepolations/', render_gif=False):
    """Linear interpolation between two images in latent space."""
    model.eval()

    #obtain latents
    z1 = model.forward(img1)[2]
    z2 = model.forward(img2)[2]

    #interpolate
    samples = []
    for n in range(1, steps + 1):
        z_new = []
        for i in range(len(z1)):
            z_new.append(z1[i] + n*(z2[i] - z1[i])/steps)
        img = model.reverse(z_new)
        samples.append(img.cpu().detach())
    samples = torch.cat(samples, dim=0)

    if filename is not None:
        os.makedirs('./sample_intepolations', exist_ok=True)
        if render_gif:
            save_gif(samples, save_path + filename + '.gif')
        else:
            TVutils.save_image(samples, save_path + filename +'.png', normalize=True, nrow=steps)
