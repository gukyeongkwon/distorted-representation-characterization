import torch

import numpy as np
from PIL import Image
import os
import shutil
from collections import OrderedDict


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpointdir):
    fullpath = os.path.join(checkpointdir, 'checkpoint.pth.tar')
    fullpath_best = os.path.join(checkpointdir, 'model_best.pth.tar')
    torch.save(state, fullpath)

    if is_best:
        shutil.copyfile(fullpath, fullpath_best)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def cal_nsample_perclass(inlier_class, outlier_class, nsamples_fold):
    """
    Calculate the number of samples for each of in-/out-of-distribution classes in a fold. The number of samples from
    in-distribution and out-of-distribution should be equal.
    Args:
        inlier_class (list): Classes for in-distribution
        outlier_class (list): Classes for out-of-distribution
        nsamples_fold (list): Number of samples for each classes in a fold
    Returns:
        cls (list): Classes for in-/out-of-distribution
        nsamples_per_cls (list): Number of samples for each of in-/out-of-distribution classes in a fold
    """

    nin = [nsamples_fold[i] for i in inlier_class]
    nout = [nsamples_fold[i] for i in outlier_class]

    if sum(nin) > sum(nout):
        small_class = outlier_class
        nsmall = nout
        large_class = inlier_class
    else:
        small_class = inlier_class
        nsmall = nin
        large_class = outlier_class

    nlarge = [sum(nsmall) // len(large_class)] * len(large_class)
    rem_class = sum(nsmall) % len(large_class)
    for cls_idx in range(0, rem_class):
        nlarge[cls_idx] += 1

    cls = small_class + large_class
    nsamples_per_cls = nsmall + nlarge

    return cls, nsamples_per_cls


def unwrap_parallel(state_dict):
    """
    Load state_dict from pretrained Dataparallel module and unwarp the Dataparallel module
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict
