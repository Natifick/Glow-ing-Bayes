import torch
import torch.nn

import torchvision.utils as TVutils

@torch.no_grad()
def get_latent_direction(model, data_loader, binary_features, attr):
    """
    Sampling from trained model based of direction in latent space.
    Direction is obtained following way:
    - select binary attribute of interest a
    - divide dataset into two subsets: a=0 and a=1
    - compute mean of latent space representation for each subset
    - obtain direction as a difference between means
    - linearly intepolate from one image two another
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
def sample_direction(model, img, direction, steps, step_size, filename=None, save_path='./sample_directions'):
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
        torch.save(samples, save_path + '.pt')
        TVutils.save_image(samples, save_path + '.png', normalize=True, nrow=steps)

    return samples


@torch.no_grad()
def sample_linear_interpolation(model, img1, img2, steps, filename=None, save_path='./sample_intepolations'):
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
        torch.save(samples, save_path + '.pt')
        TVutils.save_image(samples, save_path + '.png', normalize=True, nrow=steps)
    
    return samples

        