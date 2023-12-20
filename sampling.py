import torch
import torch.nn

import torchvision.utils as TVutils

import os
import imageio


def save_gif(images: torch.Tensor, path: str):
    images = images.cpu().detach().numpy()
    images = [image.transpose(1, 2, 0) for image in images]
    imageio.mimsave(path, images, fps=10)


@torch.no_grad()
def get_latent_direction(model, data_loader, binary_features, attr):
    """
    Obtain direction from {no attr} to {attr} in latent space
    """
    model.eval()
    cum_mean_from = torch.zeros(1, model.latent_dim)
    cum_mean_to = torch.zeros(1, model.latent_dim)

    n_from, n_to = 0, 0
    for i, img in enumerate(data_loader):
        img = img.to(model.device)
        if binary_features[i][attr] == 0:
            cum_mean_from = (model.forward(img)[2] + n_from*cum_mean_from) / (n_from + 1)
            n_from += 1
        else:  # binary_features[i][attr] == 1
            cum_mean_to += (model.forward(img)[2] + n_to*cum_mean_to) / (n_to + 1)
            n_to += 1

    return cum_mean_to - cum_mean_from


@torch.no_grad()
def sample_direction(model, img, direction, steps, step_size, filename=None, save_path='./sample_directions/', render_gif=False):
    """Sample in direction in latent space."""
    model.eval()
    # obtain latent
    img = img.to(model.device)
    z = model.forward(img)[2]

    # sample in direction
    samples = []
    for n in range(1, steps + 1):
        z_shifted = z + n*step_size*direction
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
def sample_linear_interpolation(model, img1, img2, steps, filename=None, save_path='./sample_intepolations', render_gif=False):
    """Linear interpolation between two images in latent space."""
    model.eval()

    #obtain latents
    img1 = img1.to(model.device)
    img2 = img2.to(model.device)

    z1 = model.forward(img1)[2]
    z2 = model.forward(img2)[2]

    #interpolate
    samples = []
    for n in range(1, steps + 1):
        z = z1 + n*(z2 - z1)/steps
        img = model.reverse(z)
        samples.append(img.cpu().detach())
    samples = torch.cat(samples, dim=0)

    if filename is not None:
        os.makedirs('./sample_interpolation', exist_ok=True)
        if render_gif:
            save_gif(samples, save_path + filename + '.gif')
        else:
            TVutils.save_image(samples, save_path + filename +'.png', normalize=True, nrow=steps)
    

        