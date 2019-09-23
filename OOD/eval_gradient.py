import torch
import torch.utils.data
import torchvision.transforms as transforms

import argparse
import os
import models
import datasets
import time
import numpy as np

import gradient
import d_gradient_train


parser = argparse.ArgumentParser(description='PyTorch Autoencoder Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--write-freq', '-wf', default=5, type=int,
                    metavar='N', help='write frequency (default: 5)')


def main():

    global args
    args = parser.parse_args()

    batch_size = 1
    dataset_dir = '/home/gukyeong/dataset/mnist/folds/28_28'
    vae_ckpt = './checkpoints/mnist/vae/gradient/' \
               'vae_BCE_gradient_reducedCnnSeq-4layer_train-fold234-0-5_val-fold1-0-5/model_best.pth.tar'
    gradient_layer = 'down_0'
    d_ckpt = './checkpoints/mnist/d/gradient/d_BCE_ShallowLinear_%s_grad_train-fold234-0-5_val-fold1-0-5_' % gradient_layer + \
              vae_ckpt.split('/')[-2] + '/model_best.pth.tar'

    # seed = random.randint(1, 100000)
    # torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = models.VAEReducedCNN()
    vae = torch.nn.DataParallel(vae).to(device)
    if os.path.isfile(vae_ckpt):
        print("=> loading checkpoint '{}'".format(vae_ckpt))
        checkpoint_vae = torch.load(vae_ckpt)
        best_loss = checkpoint_vae['best_loss']
        vae.load_state_dict(checkpoint_vae['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
              .format(vae_ckpt, checkpoint_vae['epoch'], best_loss))
    else:
        print("=> no checkpoint found at '{}'".format(vae_ckpt))

    ngradlayer = gradient_layer.split('_')[1]
    if gradient_layer.split('_')[0] == 'down':
        grad_dim = vae.module.down[int(ngradlayer)].weight.view(-1).shape[0]
    else:
        grad_dim = vae.module.down[int(ngradlayer)].weight.view(-1).shape[0]
    d = models.DisShallowLinear(grad_dim)
    d = torch.nn.DataParallel(d).to(device)
    d.eval()

    if os.path.isfile(d_ckpt):
        print("=> loading checkpoint '{}'".format(d_ckpt))
        checkpoint_d = torch.load(d_ckpt)
        best_acc = checkpoint_d['best_acc']
        d.load_state_dict(checkpoint_d['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}, best_acc {})"
              .format(d_ckpt, checkpoint_d['epoch'], best_acc))
    else:
        print("=> no checkpoint found at '{}'".format(d_ckpt))

    # Load dataset
    # # Make the number of inlier samples and outlier samples equal
    # inlier_class = [0, 1, 2, 3, 4, 6, 7, 8, 9]
    # class_of_interest = 5
    # # test_nsamples_perclass = utils.cal_nsample_perclass([class_of_interest], 890)
    # # test_nsamples_perclass = [0] * 10
    # test_nsamples_perclass = [445, 445, 445, 445, 445, 445, 0, 890, 890, 890]
    # # test_nsamples_perclass = [149, 149, 148, 148, 148, 148, 890, 0, 0, 0]
    # batch_size = 1
    #
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.SubsetMNISTdataset(dataset_dir, train=False, transform=transforms.ToTensor(),
    #                                 target_transform= transforms.ToTensor(),
    #                                 nsamples_perclass=test_nsamples_perclass),
    #     batch_size=batch_size, shuffle=False)

    nsamples_fold = [1380, 1575, 1398, 1428, 1364, 1262, 1375, 1458, 1365, 1391]
    inlier_class = [0, 1, 2, 3, 4, 5]
    outlier_class = [7, 8, 9]
    cls = inlier_class + outlier_class

    nout = []
    for n in outlier_class:
        nout.append(nsamples_fold[n])

    total_nout = sum(nout)
    nin = [total_nout // len(inlier_class)] * len(inlier_class)
    rem_class = total_nout % len(inlier_class)
    for cls_idx in range(0, rem_class):
        nin[cls_idx] += 1

    test_loader = torch.utils.data.DataLoader(
        datasets.FoldMNISTdataset(dataset_dir, folds=[0],
                                  transform=transforms.ToTensor(),
                                  target_transform=transforms.ToTensor(),
                                  cls=cls, nsamples_per_cls=nin + nout),
        batch_size=batch_size, shuffle=True)

    # Start evaluation
    vbackprop = gradient.VanillaBackprop(vae)
    timestart = time.time()
    print('\n*** Start Testing *** \n')
    _, _, result = d_gradient_train.test(d, vbackprop, [7, 8, 9], gradient_layer, device, test_loader, 1, args.print_freq)
    print('Total processing time: %.4f' % (time.time() - timestart))

    # inliers
    X1 = [x[1] for x in result if x[0] == 1]

    # outliers
    Y1 = [x[1] for x in result if not x[0] == 0]

    min_delta = min([x[1] for x in result]) - 0.05
    max_delta = max([x[1] for x in result]) + 0.05

    ##################################################################
    # FPR at TPR 95
    ##################################################################
    fpr95 = 0.0
    clothest_tpr = 1.0
    dist_tpr = 1.0
    for e in np.arange(min_delta, max_delta, 0.05):
        tpr = np.sum(np.greater_equal(X1, e)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
        if abs(tpr - 0.95) < dist_tpr:
            dist_tpr = abs(tpr - 0.95)
            clothest_tpr = tpr
            fpr95 = fpr

    print("tpr: ", clothest_tpr)
    print("fpr95: ", fpr95)

    ##################################################################
    # Detection error
    ##################################################################
    detect_error = 1.0
    for e in np.arange(min_delta, max_delta, 0.05):
        tpr = np.sum(np.less(X1, e)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1, e)) / np.float(len(Y1))
        detect_error = np.minimum(detect_error, (tpr + fpr) / 2.0)

    print("Detection error: ", detect_error)

    ##################################################################
    # AUPR IN
    ##################################################################
    auprin = 0.0
    recallTemp = 1.0
    for e in np.arange(min_delta, max_delta, 0.05):
        tp = np.sum(np.greater_equal(X1, e))
        fp = np.sum(np.greater_equal(Y1, e))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin += (recallTemp - recall) * precision
        recallTemp = recall
    auprin += recall * precision

    print("auprin: ", auprin)

    ##################################################################
    # AUPR OUT
    ##################################################################
    minp, max_delta = -max_delta, -min_delta
    X1 = [-x for x in X1]
    Y1 = [-x for x in Y1]
    auprout = 0.0
    recallTemp = 1.0
    for e in np.arange(min_delta, max_delta, 0.05):
        tp = np.sum(np.greater_equal(Y1, e))
        fp = np.sum(np.greater_equal(X1, e))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprout += (recallTemp - recall) * precision
        recallTemp = recall
    auprout += recall * precision

    print("auprout: ", auprout)

    # with open(
    #         os.path.join("results_train_%d-%d_val_%d.txt" % (inliner_classes[0], inliner_classes[-1], val_classes[0])),
    #         "w") as file:
    #     file.write(
    #         "\nF1: %f \nAUC: %f \nfpr95: %f"
    #         "\nDetection Error: %f \nauprin: %f \nauprout: %f\n\n" %
    #         (f1, auc, fpr95, detect_error, auprin, auprout))


if __name__ == '__main__':
    main()
