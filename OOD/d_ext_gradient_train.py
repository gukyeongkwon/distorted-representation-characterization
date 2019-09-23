import torch
import torch.nn.functional as func

import utils
import numpy as np


def train(d, outlier_classes, device, in_loader, optimizer, epoch, print_freq, out_iter=None):
    """
    Args:
        d (DataParallel(module)): discriminator
        outlier_classes (list): A list of classes considered as out-of-distribution classes
        device (Device): CPU or GPU
        in_loader (DataLoader): data loader for in-distribution data
        optimizer
        epoch (int)
        print_freq(int): Determine how often to print out progress
        out_iter (iter): Iterator for loading out-of-distribution data
    """
    d.train()
    losses = utils.AverageMeter()
    correct = 0

    for batch_idx, (features, class_label) in enumerate(in_loader):

        if out_iter is None:
            data = features.to(device)
            inout_label = torch.ones([features.shape[0]], dtype=torch.float32).to(device)  # in: 1, out: 0
            for outlier_class in outlier_classes:
                inout_label -= (class_label == outlier_class).type(torch.float32).to(device)
        else:
            out_features, out_labels = out_iter.next()
            data = torch.cat((features, out_features), dim=0)
            inout_label = torch.cat((class_label, out_labels), dim=0).to(device)

        optimizer.zero_grad()
        output = d(data)

        loss = func.binary_cross_entropy(output[:, 0], inout_label)
        losses.update(loss.item(), data.size(0))
        loss.backward()
        optimizer.step()
        pred = output >= 0.5
        correct += pred.eq(inout_label.type(torch.uint8).view_as(pred)).sum().item()

        if batch_idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batch_idx, len(in_loader), loss=losses))

    if out_iter is None:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            losses.val, correct, len(in_loader.dataset), 100. * correct / (len(in_loader.dataset))))
    else:
        print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
            losses.val, correct, len(in_loader.dataset) * 2, 100. * correct / (len(in_loader.dataset) * 2)))


def test(d, outlier_classes, device, in_loader, epoch, print_freq, out_iter=None, is_eval=False):
    """
    Args:
        d (DataParallel(module)): discriminator
        outlier_classes (list): A list of classes considered as out-of-distribution classes
        device (Device): CPU or GPU
        in_loader (DataLoader): data loader for in-distribution data
        epoch (int)
        print_freq(int): Determine how often to print out progress
        out_iter (iter): Iterator for loading out-of-distribution data
        is_eval (bool): If true, the output of the discriminator and labels are recorded for the performance evaluation
    """
    d.eval()
    losses = utils.AverageMeter()
    correct = 0
    cnt = 0
    if out_iter is None:
        nsamples = len(in_loader.dataset)
    else:
        nsamples = len(in_loader.dataset) * 2
    results = np.zeros([nsamples, 2])
    for batch_idx, (features, class_label) in enumerate(in_loader):

        if out_iter is None:
            data = features.to(device)
            inout_label = torch.ones([features.shape[0]], dtype=torch.float32).to(device)
            for outlier_class in outlier_classes:
                inout_label -= (class_label == outlier_class).type(torch.float32).to(device)
        else:
            out_features, out_labels = out_iter.next()
            data = torch.cat((features, out_features), dim=0)
            inout_label = torch.cat((class_label, out_labels), dim=0).to(device)

        output = d(data)
        loss = func.binary_cross_entropy(output[:, 0], inout_label)
        losses.update(loss.item(), data.size(0))

        pred = output >= 0.5
        correct += pred.eq(inout_label.type(torch.uint8).view_as(pred)).sum().item()

        if is_eval:
            results[cnt:cnt + data.shape[0], :] \
                = np.vstack(
                [inout_label.cpu().detach().numpy(), output.cpu().detach().numpy()[:, 0]]).transpose()
            cnt += data.shape[0]

        if batch_idx % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, batch_idx, len(in_loader), loss=losses))

    accuracy = 100. * correct / nsamples
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        losses.val, correct, nsamples, accuracy))

    return losses.avg, accuracy, results
