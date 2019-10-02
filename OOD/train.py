import torch
import torchvision.utils as vutils
import torch.utils.data
import torchvision.transforms as transforms
from torch import optim

import argparse
import os
import time
import random
from tensorboardX import SummaryWriter

import models
import utils
import datasets
import vae_train


parser = argparse.ArgumentParser(description='VAE Training')
parser.add_argument('-e', '--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--write-freq', '-wf', default=5, type=int,
                    metavar='N', help='write frequency (default: 5)')
parser.add_argument('-r', '--resume', action='store_true', help='resume training from a checkpoint')
parser.add_argument('--write-enable', '-we', action='store_true', help='enable writing')
parser.add_argument('--dataset-dir', default='./cure-tsr/RealChallengeFree', type=str, help='dataset directory')


def main():
    
    global args
    args = parser.parse_args()

    savedir = 'vae/vae_BCE_gradient_reducedCnnSeq-4layer_train-00_00_val-00_00'
    checkpointdir = os.path.join('./checkpoints', savedir)
    logdir = os.path.join('./logs', savedir)

    seed = random.randint(1, 100000)
    torch.manual_seed(seed)
    dataset_dir = args.dataset_dir

    if args.write_enable:
        os.makedirs(checkpointdir)
        writer = SummaryWriter(log_dir=logdir)
        print('log directory: %s' % logdir)
        print('checkpoints directory: %s' % checkpointdir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = models.VAECURECNN()
    vae = torch.nn.DataParallel(vae).to(device)
    best_score = 1e20
    batch_size = 128
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    if args.resume:
        vae_resume_ckpt = './checkpoints/cure-tsr/vae/' \
                          'vae_BCE_gradient_reducedCnnSeq-4layer_train-00_00_val-00_00/model_best.pth.tar'
        if os.path.isfile(vae_resume_ckpt):
            print("=> loading checkpoint '{}'".format(vae_resume_ckpt))
            checkpoint = torch.load(vae_resume_ckpt)
            vae.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                  .format(vae_resume_ckpt, checkpoint['epoch'], checkpoint['best_loss']))
        else:
            print("=> no checkpoint found at '{}'".format(vae_resume_ckpt))

    in_train_loader = torch.utils.data.DataLoader(
        datasets.CURETSRdataset(os.path.join(dataset_dir, 'train'),
                                transform=transforms.Compose([transforms.Resize([28, 28]),
                                                              transforms.ToTensor()]),
                                target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                     transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True)

    in_val_loader = torch.utils.data.DataLoader(
        datasets.CURETSRdataset(os.path.join(dataset_dir, 'val'),
                                transform=transforms.Compose([transforms.Resize([28, 28]),
                                                              transforms.ToTensor()]),
                                target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                     transforms.ToTensor()])),
        batch_size=batch_size, shuffle=True)

    # Start training
    timestart = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        print('\n*** Start Training *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs))
        vae_train.train(vae, device, in_train_loader, optimizer, epoch + 1, args.print_freq)
        print('\n*** Start Testing *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs))
        loss, input_img, recon_img, target_img = vae_train.test(vae, device, in_val_loader, epoch + 1,
                                                                args.print_freq)

        is_best = loss < best_score
        best_score = min(loss, best_score)

        if is_best:
            best_epoch = epoch + 1

        if args.write_enable:
            if epoch % args.write_freq == 0:
                writer.add_scalar('loss', loss, epoch + 1)
                writer.add_image('input_img', vutils.make_grid(input_img, nrow=3), epoch + 1)
                writer.add_image('recon_img', vutils.make_grid(recon_img, nrow=3), epoch + 1)
                writer.add_image('target_img', vutils.make_grid(target_img, nrow=3), epoch + 1)

                utils.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': vae.state_dict(),
                    'best_loss': best_score,
                    'last_loss': loss,
                    'optimizer': optimizer.state_dict(),
                }, is_best, checkpointdir)

    if args.write_enable:
        writer.close()

    print('Best Testing Acc/Loss: %.3f at epoch %d' % (best_score, best_epoch))
    print('Best epoch: ', best_epoch)
    print('Total processing time: %.4f' % (time.time() - timestart))


if __name__ == '__main__':
    main()
