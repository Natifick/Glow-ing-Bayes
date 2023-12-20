import torch
import torch.nn

import torchvision.utils as TVutils

"""
Code responsible for sampling from trained model based of direction in latent space.
Directions are obtained following way:
- select binary attribute of interest a
- divide dataset into two subsets: a=0 and a=1
- compute mean of latent space representation for each subset
- obtain direction as a difference between means
- linearly intepolate from one image two another
"""

@torch.no_grad()
def get_latent_direction(model, data_loader, binary_features, attr):
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
def get_linear_interpolation(model, img, direction, steps, step_size, filename=None, save_path='./sample_interpolation'):
    model.eval()
    img = img.to(model.device)
    z = model.forward(img)[2]
    samples = []
    for n in range(1, steps + 1):
        z_shifted = z + n*step_size*direction
        img_shifted = model.reverse(z_shifted)
        samples.append(img_shifted.cpu().detach())
    samples = torch.cat(samples, dim=0)
    
    if filename is not None:
        torch.save(samples, save_path + '.pt')
        TVutils.save_image(samples, save_path + '.png', normalize=True, nrow=steps)

    return samples
        