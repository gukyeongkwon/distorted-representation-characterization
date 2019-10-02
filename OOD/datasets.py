import torch
import torch.utils.data as data

import os
from PIL import Image


class GradDataset(data.Dataset):

    def __init__(self, path_list, transform=None):
        """
        Args:
            path_list (list): A list of paths for gradient data extracted from different layers or losses
            transform (class): Transform applied to the input data
        """
        self.path_list = path_list
        self.transform = transform
        self.label_grad_data = None

        for idx, path in enumerate(path_list):
            if idx == 0:
                with open(path, 'rb') as pkl:
                    label, grad = torch.load(pkl)
                    grad_all = grad
                    # grad_all = grad / grad.norm(dim=1).view(-1, 1)
            else:
                with open(path, 'rb') as pkl:
                    tmp_grad = torch.load(pkl)[1]
                    # tmp_grad = grad / grad.norm(dim=1).view(-1, 1)
                    grad_all = torch.cat((grad_all, tmp_grad), dim=1)
        self.label_grad_data = (label, grad_all)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            grad_data (tensor): gradient features (dim: (number of images) x (dim. of gradients))
            label (tensor):  Label for input image
        """
        labels, grad_data = self.label_grad_data[0][index], self.label_grad_data[1][index, :]

        if self.transform is not None:
            grad_data = self.transform(grad_data)

        return grad_data, labels

    def __len__(self):
        return self.label_grad_data[0].shape[0]


class CURETSRdataset (data.Dataset):
    def __init__(self, datadir, transform=None, target_transform=None):
        """
        Args:
            datadir (str): Directory for CURE-TSR dataset
            transform (class): Transform applied to the input
            target_transform (class): Transform applied to the target image of reconstruction
        """
        self.datadir = datadir
        self.transform = transform
        self.target_transform = target_transform
        self.img_label_data = []

        for fname in sorted(os.listdir(self.datadir)):
            if int(fname.split('_')[2]) == 0:
                label = 1  # Challenge image
            else:
                label = 0  # Challenge-free image
            img = Image.open(os.path.join(self.datadir, fname)).convert('RGB')
            self.img_label_data.append((img, label))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img (tensor): Input image after transform
            target (tensor): Target image after transform for the reconstruction
            label (tensor): Label for input image
        """
        img, label = self.img_label_data[index][0], self.img_label_data[index][1]
        target = img

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label

    def __len__(self):
        return len(self.img_label_data)