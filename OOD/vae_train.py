import torch
import torch.nn.functional as func

import numpy as np

import utils


def gaussian_loss_function(mu_decode, logvar_decode, x, mu, logvar, beta=1.0, imgsize=784, size_average=False):
    """
    Args:
        mu_decode (tensor): reconstructed mean from VAE (dim: (batch size) x (channel) x (height) x (width))
        logvar_decode (tensor): reconstructed logvar from VAE (dim: (batch size) x (channel) x (height) x (width))
        x (tensor): Input images to VAE (dim: (batch size) x (channel) x (height) x (width))
        mu (tensor): Estimated mean of Gaussian in latent space (dim: (batch size) x (dim. of latent space)
        logvar (tensor): Estimated logvar of Gaussian in latent space (dim: (batch size) x (dim. of latent space)
        beta (int): Weight on latent constraint
        imgsize (int): height x width
        size_average (bool): If true, loss will be averaged. Otherwize, loss will be summed.
    Returns:
        loss (tensor): loss calculated by taking the sum of all losses from images in the batch
    """
    mean = 0
    std = 1

    target_mu = (mean * torch.ones(mu.size())).cuda()
    target_logvar = (np.log(std**2) * torch.ones(logvar.size())).cuda()

    gaussian_prob = torch.distributions.Normal(mu_decode.view(-1, imgsize),
                                               logvar_decode.view(-1, imgsize).exp().sqrt())
    log_gauss = gaussian_prob.log_prob(x.view(-1, imgsize))
    log_gauss = log_gauss.sum() * (-1)
    kld = -0.5 * torch.sum(1 + (logvar - target_logvar) - 1
                           * ((target_mu - mu).pow(2) + logvar.exp()).div(target_logvar.exp()))

    if size_average:
        log_gauss /= (x.size()[0] * imgsize)
        kld /= (x.size()[0] * imgsize)

    return log_gauss + beta * kld


def bernoulli_loss_function(recon_x, x, mu, logvar, beta=1.0, imgsize=784, loss_type=None):
    """
    Args:
        recon_x (tensor): reconstructed images from VAE (dim: (batch size) x (channel) x (height) x (width))
        x (tensor): Input images to VAE (dim: (batch size) x (channel) x (height) x (width))
        mu (tensor): Estimated mean of Gaussian in latent space (dim: (batch size) x (dim. of latent space)
        logvar (tensor): Estimated logvar of Gaussian in latent space (dim: (batch size) x (dim. of latent space)
        beta (int): Weight on latent constraint
        imgsize (int): height x width
        loss_type (str): Choose type of loss to be returned
    Returns:
        loss (tensor): loss calculated by taking the sum of all losses from images in the batch
    """
    bce = func.binary_cross_entropy(recon_x.view(-1, imgsize), x.view(-1, imgsize), size_average=False)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if loss_type is None:
        loss = bce + beta * kld
    elif loss_type == 'bce':
        loss = bce
    elif loss_type == 'kld':
        loss = kld
    return loss


def train(model, device, train_loader, optimizer, epoch, print_freq):
    """
    Args:
         model (DataParallel(module)): VAE
         device (Device): CPU or GPU
         train_loader (DataLoader): data loader for training data
         optimizer
         epoch (int)
         print_freq(int): Determine how often to print out progress
    """
    model.train()
    losses = utils.AverageMeter()
    for batch_idx, (data, target_data, label) in enumerate(train_loader):
        data = data.to(device)
        target_data = target_data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)
        loss = bernoulli_loss_function(recon_batch, target_data, mu, logvar, beta=1,
                                       imgsize=recon_batch.view(data.data.size(0), -1).size(1))

        losses.update(loss.item(), data.size(0))  # data.size(0): Batch size
        loss.backward()

        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batch_idx, len(train_loader), loss=losses))


def test(model, device, test_loader, epoch, print_freq):
    """
    Args:
         model (DataParallel(module)): VAE
         device (Device): CPU or GPU
         test_loader (DataLoader): data loader for test data
         epoch (int)
         print_freq(int): Determine how often to print out progress
    """
    model.eval()
    losses = utils.AverageMeter()

    with torch.no_grad():
        for batch_idx, (data, target_data, label) in enumerate(test_loader):
            data = data.to(device)
            target_data = target_data.to(device)

            #  dSprites, MNIST dataset
            recon_batch, mu, logvar = model(data)
            loss = bernoulli_loss_function(recon_batch, target_data, mu, logvar, beta=1,
                                           imgsize=recon_batch.view(data.data.size(0), -1).size(1))
            losses.update(loss.item(), data.size(0))

            if batch_idx == 0:
                nimg = 3
                input_img = data[:nimg]
                recon_img = recon_batch[:nimg, :].view(nimg, data.shape[1], data.shape[2], data.shape[3])
                target_img = target_data[:nimg]

            if batch_idx % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, batch_idx, len(test_loader), loss=losses))

    print(' * Loss {loss.avg:.3f}'.format(loss=losses))
    return losses.avg, input_img, recon_img, target_img
