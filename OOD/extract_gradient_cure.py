import torch
import torch.utils.data
import torchvision.transforms as transforms

import argparse
import os
import models
import datasets
import time
import errno

import gradient
import utils


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

    root_dir = '/media/gukyeong/HardDisk/dataset/CURE-TSR/folds'
    batch_size = 1
    vae_ckpt = './checkpoints/vae/' \
               'vae_BCE_gradient_reducedCnnSeq-4layer_train-14_val-14/model_best.pth.tar'

    sets = ['train', 'val', 'test']
    vae_layers = ['up_6']
    challenges = ['00_00']
    nsamples_fold = [325, 71, 17, 10, 235, 26, 295, 59, 20, 101, 16, 31, 179, 735]  # 14 classes
    in_cls = [13]
    out_cls = [i for i in range(13)]
    cls, nsamples_per_cls = utils.cal_nsample_perclass(in_cls, out_cls, nsamples_fold)

    # for challID in range(1, 13):
    #     for levelID in range(1, 6):
    #         challenges.append('%02d_%02d' % (challID, levelID))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = models.VAECURECNN()
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

    for chall in challenges:
        if chall == '00_00':
            dataset_dir = os.path.join(root_dir, 'RealChallengeFree')
        else:
            dataset_dir = os.path.join(root_dir, 'RealChallenge/%s' % chall)
        for gradient_layer in vae_layers:
            for set_name in sets:
                print('Extracting %s chall %s layer %s set\n' % (chall, gradient_layer, set_name))

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

                if set_name == 'train':
                    cls = in_cls
                    data_loader = torch.utils.data.DataLoader(
                        datasets.CURETSRdataset(os.path.join(dataset_dir, 'train'),
                                                transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                              transforms.ToTensor()]),
                                                target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                                     transforms.ToTensor()]),
                                                cls=cls),
                        batch_size=batch_size, shuffle=True)

                elif set_name == 'val':
                    cls = in_cls
                    data_loader = torch.utils.data.DataLoader(
                        datasets.CURETSRdataset(os.path.join(dataset_dir, 'val'),
                                                transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                              transforms.ToTensor()]),
                                                target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                                     transforms.ToTensor()]),
                                                cls=cls),
                        batch_size=batch_size, shuffle=True)

                elif set_name == 'test':
                    cls = in_cls + out_cls
                    data_loader = torch.utils.data.DataLoader(
                        datasets.CURETSRdataset(os.path.join(dataset_dir, 'test'),
                                                transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                              transforms.ToTensor()]),
                                                target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                                     transforms.ToTensor()]),
                                                cls=cls, nsamples_per_cls=nsamples_per_cls),
                        batch_size=batch_size, shuffle=True)

                # Start evaluation
                vbackprop = gradient.VanillaBackprop(vae, loss_type='bce')
                timestart = time.time()
                print('\n*** Start Testing *** \n')

                labels = torch.zeros(len(data_loader),)
                grad_data = torch.zeros(len(data_loader), grad_dim)
                for batch_idx, (input_img, target_img, class_label) in enumerate(data_loader):
                    print('Processing...(%d / %d)' % (batch_idx, len(data_loader)))
                    input_img = input_img.to(device)
                    target_img = target_img.to(device)

                    vbackprop.generate_gradients(input_img, target_img)
                    grad_data[batch_idx, :] = vbackprop.gradients[gradient_layer].view(1, -1)
                    labels[batch_idx] = class_label

                savedir = os.path.join(root_dir, 'in-14_out-others', vae_ckpt.split('/')[-2])
                try:
                    os.makedirs(savedir)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise

                if set_name == 'test':
                    torch.save((labels, grad_data), os.path.join(savedir, '%s_incls%s_outcls%s_%s.pt'
                                                                 % (set_name,
                                                                    ''.join(str(i) for i in in_cls),
                                                                    ''.join(str(i) for i in out_cls),
                                                                    gradient_layer)))
                else:
                    torch.save((labels, grad_data), os.path.join(savedir, '%s_cls%s_%s.pt'
                                                                 % (set_name,
                                                                    ''.join(str(i) for i in cls), gradient_layer)))

                print('Total processing time: %.4f' % (time.time() - timestart))


if __name__ == '__main__':
    main()
