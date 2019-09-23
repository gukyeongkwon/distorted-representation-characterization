import torch
import torch.nn.functional as func

import numpy as np

import utils


def train(d, device, in_loader, optimizer, epoch, print_freq, vae=None, std=1.0, out_iter=None, use_reconimg=False):
    """
    Args:
        d (DataParallel(module)): discriminator
        device (Device): CPU or GPU
        in_loader (DataLoader): data loader for in-distribution data
        optimizer
        epoch (int)
        print_freq(int): Determine how often to print out progress
        vae (DataParallel(module)): vae
        std (float): standard deviation for sampling from Gaussian distribution
        out_iter (iter): Iterator for loading out-of-distribution data
        use_reconimg (bool): Use reconstructed image for training the discriminator if true
    """
    d.train()
    losses = utils.AverageMeter()
    correct = 0
    for batch_idx, (in_data, _, _) in enumerate(in_loader):
        # Generated out-of-distribution samples
        if out_iter is None:
            vae.eval()
            out_z = torch.randn([in_data.shape[0], 20]).to(device) * std
            out_data = vae.module.decode(out_z)
            out_data = out_data.view_as(in_data)
            if use_reconimg:
                in_data, _, _ = vae(in_data)
        else:
            out_data, _, _ = out_iter.next()

        if vae is None:
            data = torch.cat((in_data, out_data), dim=0)
            target = torch.cat((torch.ones(in_data.shape[0], dtype=torch.float32),
                                torch.zeros(out_data.shape[0], dtype=torch.float32))).to(device)
            output = d(data.view(data.shape[0], -1))
        else:
            # in/out-of-distribution latent variables
            in_data = in_data.to(device)
            mu, logvar = vae.module.encode(in_data)
            in_z = vae.module.reparameterize(mu, logvar)

            out_data = out_data.to(device)
            mu, logvar = vae.module.encode(out_data)
            out_z = vae.module.reparameterize(mu, logvar)

            data = torch.cat((in_z, out_z), dim=0)
            target = torch.cat((torch.ones(in_z.shape[0], dtype=torch.float32),
                                torch.zeros(in_z.shape[0],dtype=torch.float32))).to(device)
            output = d(data)

        optimizer.zero_grad()
        loss = func.binary_cross_entropy(output[:, 0], target)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()
        pred = output >= 0.5
        correct += pred.eq(target.type(torch.uint8).view_as(pred)).sum().item()

        if batch_idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batch_idx, len(in_loader), loss=losses))

    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        losses.val, correct, len(in_loader.dataset) * 2, 100. * correct / (len(in_loader.dataset) * 2)))


def test(d, device, in_loader, epoch, print_freq, vae=None, std=1.0, out_iter=None, use_reconimg=False, is_eval=False):
    """
    Args:
        d (DataParallel(module)): discriminator
        device (Device): CPU or GPU
        in_loader (DataLoader): data loader for in-distribution data
        epoch (int)
        print_freq(int): Determine how often to print out progress
        vae (DataParallel(module)): vae
        std (float): standard deviation for sampling from Gaussian distribution
        out_iter (iter): Iterator for loading out-of-distribution data
        use_reconimg (bool): Use reconstructed image for training the discriminator if true
        is_eval (bool): If true, the output of the discriminator and label are recorded for the performance evaluation
    """
    d.eval()
    losses = utils.AverageMeter()
    correct = 0
    cnt = 0
    results = np.zeros([len(in_loader.dataset) * 2, 2])

    with torch.no_grad():
        for batch_idx, (in_data, _, _) in enumerate(in_loader):
            # Generated out-of-distribution samples
            if out_iter is None:
                vae.eval()
                out_z = torch.randn([in_data.shape[0], 20]).to(device) * std
                out_data = vae.module.decode(out_z)
                out_data = out_data.view_as(in_data)
                if use_reconimg:
                    in_data, _, _ = vae(in_data)
            else:
                out_data, _, _ = out_iter.next()

            if vae is None:
                data = torch.cat((in_data, out_data), dim=0)
                target = torch.cat((torch.ones(in_data.shape[0], dtype=torch.float32),
                                    torch.zeros(out_data.shape[0], dtype=torch.float32))).to(device)
                output = d(data.view(data.shape[0], -1))
            else:
                # in/out-of-distribution latent variables
                in_data = in_data.to(device)
                mu, logvar = vae.module.encode(in_data)
                in_z = vae.module.reparameterize(mu, logvar)

                out_data = out_data.to(device)
                mu, logvar = vae.module.encode(out_data)
                out_z = vae.module.reparameterize(mu, logvar)

                data = torch.cat((in_z, out_z), dim=0)
                target = torch.cat((torch.ones(in_z.shape[0], dtype=torch.float32),
                                    torch.zeros(in_z.shape[0], dtype=torch.float32))).to(device)
                output = d(data)

            if is_eval:
                results[cnt:(cnt + data.shape[0]), :] \
                    = np.vstack(
                    [target.cpu().detach().numpy(), output.cpu().detach().numpy()[:, 0]]).transpose()
                cnt += data.shape[0]

            loss = func.binary_cross_entropy(output[:, 0], target)
            losses.update(loss.item(), data.size(0))
            pred = output >= 0.5
            correct += pred.eq(target.type(torch.uint8).view_as(pred)).sum().item()

            if batch_idx % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, batch_idx, len(in_loader), loss=losses))

    accuracy = 100. * correct / (len(in_loader.dataset) * 2)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        losses.val, correct, len(in_loader.dataset) * 2, accuracy))

    return losses.avg, accuracy, results
