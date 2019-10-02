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


parser = argparse.ArgumentParser(description='Evaluation of a trained discriminator.')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dataset-dir',  default='./cure-tsr', type=str, help='directory for extracted gradients')


def main():

    global args
    args = parser.parse_args()

    batch_size = 64
    train_chall = '07_01'
    chall_type_list = [1, 2, 3, 5, 8, 9]
    all_results = np.zeros([5, len(chall_type_list) * 5])

    challcnt = 0
    for challID in chall_type_list:
        for levelID in range(1, 6):
            test_chall = '%02d_%02d' % (challID, levelID)
            vae_ckpt = './checkpoints/cure-tsr/vae/' \
                       'vae_BCE_gradient_reducedCnnSeq-4layer_train-00_00_val-00_00/model_best.pth.tar'
            gradient_layer = 'down_6'  # kld
            gradient_layer2 = 'up_6'  # bce
            d_ckpt = './checkpoints/cure-tsr/d/%s/bce_kld_grad/' \
                     'd_BCE_ShallowLinear_bce-%s_kld-%s_in-00_00_out-%s/model_best.pth.tar' \
                     % (vae_ckpt.split('/')[-2], gradient_layer2, gradient_layer, train_chall)
            dataset_dir = os.path.join(args.dataset_dir, 'kld_grad/%s' % vae_ckpt.split('/')[-2])
            dataset_dir2 = os.path.join(args.dataset_dir, 'bce_grad/%s' % vae_ckpt.split('/')[-2])

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

            grad_dim = vae.module.down[6].weight.view(-1).shape[0] + vae.module.up[6].weight.view(-1).shape[0]

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
                datasets.GradDataset([os.path.join(dataset_dir, '00_00_test_%s.pt' % gradient_layer),
                                      os.path.join(dataset_dir2, '00_00_test_%s.pt' % gradient_layer2)]),
                batch_size=batch_size, shuffle=True)

            out_test_loader = torch.utils.data.DataLoader(
                datasets.GradDataset([os.path.join(dataset_dir, '%s_test_%s.pt' % (test_chall, gradient_layer)),
                                      os.path.join(dataset_dir2, '%s_test_%s.pt' % (test_chall, gradient_layer2))]),
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

    np.savetxt('./results_%s_%s.csv' % (vae_ckpt.split('/')[-2], d_ckpt.split('/')[-2]), all_results,
               fmt='%.3f', delimiter=',')


if __name__ == '__main__':
    main()
