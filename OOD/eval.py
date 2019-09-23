import torch
import torch.utils.data
import torchvision.transforms as transforms

import argparse
import os
import models
import datasets
import time
import numpy as np

import d_train
import cal_metric


parser = argparse.ArgumentParser(description='PyTorch Autoencoder Training')
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

    batch_size = 64
    train_chall = '07_01'
    all_results = np.zeros([5, 12 * 5])

    challcnt = 0
    for challID in range(1, 13):
        for levelID in range(1, 6):
            test_chall = '%02d_%02d' % (challID, levelID)
            vae_ckpt = './checkpoints/cure-tsr/vae/' \
                       'vae_BCE_gradient_reducedCnnSeq-4layer_train-00_00_val-00_00/model_best.pth.tar'
            d_ckpt = './checkpoints/cure-tsr/d/%s/d_BCE_ShallowLinear_in-00_00_out-%s' \
                     % (vae_ckpt.split('/')[-2], train_chall) + '/model_best.pth.tar'

            print('\n*** Test on Challenge %s ***\n' % test_chall)

            dataset_dir = '/media/gukyeong/HardDisk/CURE-TSR/folds/RealChallengeFree'
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            vae = models.VAECURECNN()
            vae = torch.nn.DataParallel(vae).to(device)
            vae.eval()
            if os.path.isfile(vae_ckpt):
                print("=> loading checkpoint '{}'".format(vae_ckpt))
                checkpoint_vae = torch.load(vae_ckpt)
                best_loss = checkpoint_vae['best_loss']
                vae.load_state_dict(checkpoint_vae['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                      .format(vae_ckpt, checkpoint_vae['epoch'], best_loss))
            else:
                print("=> no checkpoint found at '{}'".format(vae_ckpt))

            d = models.DisShallowLinear(20)
            d = torch.nn.DataParallel(d).to(device)
            d.eval()

            if os.path.isfile(d_ckpt):
                print("=> loading checkpoint '{}'".format(d_ckpt))
                checkpoint_d = torch.load(d_ckpt)
                best_loss = checkpoint_d['best_loss']
                d.load_state_dict(checkpoint_d['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                      .format(d_ckpt, checkpoint_d['epoch'], best_loss))
            else:
                print("=> no checkpoint found at '{}'".format(d_ckpt))

            # CURE-TSR (Train : Val : Test = 6 : 2 : 2)
            in_test_loader = torch.utils.data.DataLoader(
                datasets.CURETSRdataset(os.path.join(dataset_dir, 'test'),
                                        transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                      transforms.ToTensor()]),
                                        target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                             transforms.ToTensor()])),
                batch_size=batch_size, shuffle=True)

            out_test_loader = torch.utils.data.DataLoader(
                datasets.CURETSRdataset(os.path.join(dataset_dir[:-4], '%s' % test_chall, 'test'),
                                        transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                      transforms.ToTensor()]),
                                        target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                             transforms.ToTensor()])),
                batch_size=batch_size, shuffle=True)

            # Start evaluation
            timestart = time.time()
            print('\n*** Start Testing *** \n')

            loss, acc, result = d_train.test(d, device, in_test_loader, 1, args.print_freq, vae=vae,
                                             std=1.0, out_iter=iter(out_test_loader), is_eval=True)

            # inliers
            in_pred = [x[1] for x in result if x[0] == 1]

            # outliers
            out_pred = [x[1] for x in result if x[0] == 0]

            all_results[0, challcnt] = acc
            all_results[1::, challcnt] = cal_metric.calMetric(in_pred, out_pred)
            challcnt += 1
            print('Total processing time: %.4f' % (time.time() - timestart))

    np.savetxt('%s_%s.csv' % (vae_ckpt.split('/')[-2], d_ckpt.split('/')[-2]), all_results, fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    main()
