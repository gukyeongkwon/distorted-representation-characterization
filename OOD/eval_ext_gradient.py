import torch
import torch.utils.data

import argparse
import os
import models
import datasets
import time
import numpy as np

import d_ext_gradient_train
import cal_metric


parser = argparse.ArgumentParser(description='PyTorch Autoencoder Training')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')


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
            gradient_layer = 'down_6'  # kld
            d_ckpt = './checkpoints/cure-tsr/d/%s/kld_grad/' \
                     'd_BCE_ShallowLinear_norm_bce-%s_in-00_00_out-%s/model_best.pth.tar' \
                     % (vae_ckpt.split('/')[-2], gradient_layer, train_chall)
            dataset_dir = '/media/gukyeong/HardDisk/CURE-TSR/folds/kld_grad/%s' % vae_ckpt.split('/')[-2]

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

            if gradient_layer == 'latent':
                grad_dim = vae.module.fc11.weight.shape[0]
            elif gradient_layer == 'input':
                grad_dim = 28 * 28 * 3
            else:
                ngradlayer = gradient_layer.split('_')[1]
                if gradient_layer.split('_')[0] == 'down':
                    grad_dim = vae.module.down[int(ngradlayer)].weight.view(-1).shape[0]
                elif gradient_layer.split('_')[0] == 'up':
                    grad_dim = vae.module.up[int(ngradlayer)].weight.view(-1).shape[0]
            # grad_dim = vae.module.down[6].weight.view(-1).shape[0] + vae.module.up[6].weight.view(-1).shape[0]

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

            in_test_loader = torch.utils.data.DataLoader(
                datasets.GradDataset([os.path.join(dataset_dir, '00_00_test_%s.pt' % gradient_layer)]),
                batch_size=batch_size, shuffle=True)

            out_test_loader = torch.utils.data.DataLoader(
                datasets.GradDataset([os.path.join(dataset_dir, '%s_test_%s.pt' % (test_chall, gradient_layer))]),
                batch_size=batch_size, shuffle=True)

            # Start evaluation
            timestart = time.time()
            print('\n*** Start Testing *** \n')
            loss, acc, result = d_ext_gradient_train.test(d, None, device, in_test_loader, 1, args.print_freq,
                                                          out_iter=iter(out_test_loader), is_eval=True)
            print('Total processing time: %.4f' % (time.time() - timestart))

            # inliers
            in_pred = [x[1] for x in result if x[0] == 1]

            # outliers
            out_pred = [x[1] for x in result if x[0] == 0]

            all_results[0, challcnt] = acc
            all_results[1::, challcnt] = cal_metric.calMetric(in_pred, out_pred)
            challcnt += 1

    np.savetxt('./csv_results/act_%s_%s.csv' % (vae_ckpt.split('/')[-2], d_ckpt.split('/')[-2]), all_results,
               fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    main()
