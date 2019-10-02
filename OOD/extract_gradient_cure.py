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
parser.add_argument('--dataset-dir', default='./cure-tsr',
                    type=str, help='dataset directory')
parser.add_argument('--split', default='train', type=str, help='train split or testing split')
parser.add_argument('--loss-type', default='bce', type=str, help='Loss type for the backpropagation (bce or kld)')


def main():

    global args
    args = parser.parse_args()

    root_dir = args.dataset_dir
    batch_size = 1
    vae_ckpt = './checkpoints/cure-tsr/vae/' \
               'vae_BCE_gradient_reducedCnnSeq-4layer_train-00_00_val-00_00/model_best.pth.tar'

    if args.split == 'train':  # Extract gradients for training and validation
        sets = ['train', 'val']
        # For the reconstruction error (bce), we use graidnets from the last layer of decoder
        if args.loss_type == 'bce':
            vae_layers = ['up_6']
        # For the latent loss, we use graidnets from the last layer of encoder
        elif args.loss_type == 'kld':
            vae_layers = ['donw_6']
        challenges = ['00_00', '07_01']

    elif args.split == 'test':  # Extract gradients for testing
        sets = ['test']
        # For the reconstruction error (bce), we use graidnets from the last layer of decoder
        if args.loss_type == 'bce':
            vae_layers = ['up_6']
        # For the latent loss, we use graidnets from the last layer of encoder
        elif args.loss_type == 'kld':
            vae_layers = ['donw_6']
        challenges = ['00_00']

        # 1: Decolorization, 2: Lens Blur, 3: Codec Error, 5: Dirty Lens, 8: Noise, 9: Rain
        chall_type_list = [1, 2, 3, 5, 8, 9]
        for challID in chall_type_list:
            for levelID in range(1, 6):
                challenges.append('%02d_%02d' % (challID, levelID))

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

                data_loader = torch.utils.data.DataLoader(
                    datasets.CURETSRdataset(os.path.join(dataset_dir, '%s' % set_name),
                                            transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                          transforms.ToTensor()]),
                                            target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                                 transforms.ToTensor()])),
                    batch_size=batch_size, shuffle=True)

                # Start evaluation
                vbackprop = gradient.VanillaBackprop(vae, loss_type=args.loss_type)
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

                savedir = os.path.join(root_dir, '%s_grad' % args.loss_type, vae_ckpt.split('/')[-2])
                try:
                    os.makedirs(savedir)
                except OSError as exception:
                    if exception.errno != errno.EEXIST:
                        raise

                torch.save((labels, grad_data), os.path.join(savedir, '%s_%s_%s.pt'
                                                             % (chall, set_name, gradient_layer)))
                print('Total processing time: %.4f' % (time.time() - timestart))


if __name__ == '__main__':
    main()
