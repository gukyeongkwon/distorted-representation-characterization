import torch
import torch.utils.data as data

import os
from PIL import Image
import numpy as np
import pickle


def get_subset_mnist(image_data, image_labels, nsamples_perclass):
    """
    Args:
        image_data (tensor): Image data (dim = (number of images) x (height) x (width))
        image_labels (tensor): Class label for each input image (dim = (number of images))
        nsamples_perclass (list): Number of samples for each class (length = 10 (classes))
    Returns:
        subset_data (tensor): Image data (dim = (sum(nsamples_perclass)) x (height) x (width))
        subset_labels (tensor): Class label for each input image (dim = (sum(nsamples_perclass)))
    """
    labelcnt = np.zeros([10, ])

    subset_data = torch.zeros([sum(nsamples_perclass), 28, 28], dtype=torch.uint8)
    subset_labels = torch.zeros([sum(nsamples_perclass)], dtype=torch.long)

    cnt = 0
    for i in range(len(image_data)):

        if labelcnt[image_labels[i]] < nsamples_perclass[image_labels[i]]:
            subset_data[cnt] = image_data[i]
            subset_labels[cnt] = image_labels[i]
            labelcnt[image_labels[i]] += 1
            cnt += 1
        else:
            if np.sum(labelcnt) == sum(nsamples_perclass):
                break
            else:
                continue

    return subset_data, subset_labels


class SubsetMNISTdataset(data.Dataset):
    # MNIST dataset with the specificed number of images for each class
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, nsamples_perclass=None):
        """
        Args:
            root (str): Directory for MNIST dataset
            train (bool): True for loading training set and false for test set
            transform (class): Transform applied to the input
            transform (class): Transform applied to the target image of reconstruction
            nsamples_perclass (list): Number of samples for each class (length = 10 (classes))
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.nsamples_perclass = nsamples_perclass

        if self.train:
            self.image_data, self.image_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.image_data, self.image_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

        if self.nsamples_perclass is not None:
            self.subset_data, self. subset_labels \
                = get_subset_mnist(self.image_data, self.image_labels, self.nsamples_perclass)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img (tensor): Input image after transform
            target (tensor): Target image after transform for the reconstruction
            label (tensor): Class label for input image
        """
        if self.nsamples_perclass is None:
            img, target, label = self.image_data[index], self.image_data[index], self.image_labels[index]
        else:
            img, target, label = self.subset_data[index], self.subset_data[index], self.subset_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        target = img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label

    def __len__(self):
        if self.nsamples_perclass is None:
            return len(self.image_data)
        else:
            return len(self.subset_data)


class FoldMNISTdataset(data.Dataset):

    def __init__(self, root, folds=[0], transform=None, target_transform=None, cls=None, nsamples_per_cls=None):
        """
        Args:
            root (str): Directory for MNIST dataset folds
            folds (list): A list of folds load
            transform (class): Transform applied to the input
            transform (class): Transform applied to the target image of reconstruction
            cls (list): A list of digit classes
            nsamples_per_cls (list): Number of samples for each class in the list 'cls'
        Number of samples for each class in a fold: [1380, 1575, 1398, 1428, 1364, 1262, 1375, 1458, 1365, 1391]
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.folds = folds  # training set or test set
        self.cls = cls
        self.nsamples_per_cls = nsamples_per_cls
        self.label_img_data = []

        fold_data = []
        for fold_idx in folds:
            with open(os.path.join(self.root, 'data_fold_%d.pkl' % fold_idx), 'rb') as pkl:
                fold_data += pickle.load(pkl)

        if self.nsamples_per_cls is None:
            for x in fold_data:
                if x[0] in self.cls:
                    self.label_img_data.append(x)
        else:
            class_cnt = [0] * len(self.cls)
            for x in fold_data:
                if x[0] in self.cls and class_cnt[self.cls.index(x[0])] < nsamples_per_cls[self.cls.index(x[0])]:
                    self.label_img_data.append(x)
                    class_cnt[self.cls.index(x[0])] += 1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img (tensor): Input image after transform
            target (tensor): Target image after transform for the reconstruction
            label (tensor): Class label for input image
        """
        label, img = self.label_img_data[index][0], self.label_img_data[index][1]

        img = Image.fromarray(img, mode='L')
        target = img

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, label

    def __len__(self):
        return len(self.label_img_data)


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
                    norm_grad = grad
                    # norm_grad = grad / grad.norm(dim=1).view(-1, 1)
            else:
                with open(path, 'rb') as pkl:
                    grad = torch.load(pkl)[1]
                    tmp_grad = grad / grad.norm(dim=1).view(-1, 1)
                    norm_grad = torch.cat((norm_grad, tmp_grad), dim=1)
        self.label_grad_data = (label, norm_grad)

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
    def __init__(self, datadir, transform=None, target_transform=None, cls=None, nsamples_per_cls=None):
        """
        Args:
            datadir (str): Directory for CURE-TSR dataset
            transform (class): Transform applied to the input
            target_transform (class): Transform applied to the target image of reconstruction
            cls (list): A list of digit classes
            nsamples_per_cls (list): Number of samples for each class in the list 'cls'
        """
        self.datadir = datadir
        self.transform = transform
        self.target_transform = target_transform
        self.img_label_data = []
        self.cls = cls
        self.nsamples_per_cls = nsamples_per_cls
        self.class_cnt = [0] * len(self.cls)

        # for fname in sorted(os.listdir(self.datadir)):
        #     if int(fname.split('_')[2]) == 0:
        #         label = 1  # Challenge image
        #     else:
        #         label = 0  # Challenge-free image
        #     img = Image.open(os.path.join(self.datadir, fname)).convert('RGB')
        #     self.img_label_data.append((img, label))

        if self.nsamples_per_cls is None:
            for fname in sorted(os.listdir(self.datadir)):
                cls_label = int(fname.split('_')[-2]) - 1
                if cls_label in self.cls:
                    img = Image.open(os.path.join(self.datadir, fname)).convert('RGB')
                    self.img_label_data.append((img, cls_label))
        else:
            for fname in sorted(os.listdir(self.datadir)):
                cls_label = int(fname.split('_')[-2]) - 1
                if cls_label in self.cls and \
                        self.class_cnt[self.cls.index(cls_label)] < nsamples_per_cls[self.cls.index(cls_label)]:
                    img = Image.open(os.path.join(self.datadir, fname)).convert('RGB')
                    self.img_label_data.append((img, cls_label))
                    self.class_cnt[self.cls.index(cls_label)] += 1

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
