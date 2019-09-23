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
import d_train


parser = argparse.ArgumentParser(description='PyTorch Autoencoder Training')
parser.add_argument('-e', '--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-pf', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--write-freq', '-wf', default=5, type=int,
                    metavar='N', help='write frequency (default: 5)')
parser.add_argument('-r', '--resume', action='store_true', help='resume training from a checkpoint')
parser.add_argument('--recon', action='store_true', help='Reconstruct images using a VAE')
parser.add_argument('--gen', action='store_true', help='generate images using a VAE')
parser.add_argument('--write-enable', '-we', action='store_true', help='enable writing')


def main():
    
    global args
    args = parser.parse_args()

    dis_train = 0  # 0: vae_train, 1: d_train
    use_vae = 0  # 0: Not use vae for discriminator training 1: Use vae for discriminator training
    std = 1.0
    chall = '07_01'

    if dis_train:
        if use_vae:
            vae_ckpt = './checkpoints/cure-tsr/vae/' \
                       'vae_BCE_gradient_reducedCnnSeq-4layer_train-00_00_val-00_00/model_best.pth.tar'
            savedir = 'cure-tsr/d/%s/d_BCE_ShallowLinear_in-00_00_out-%s' \
                      % (vae_ckpt.split('/')[-2], chall)

        else:
            savedir = 'cure-tsr/d/d_BCE_ShallowLinear_in-00_00_out-%s' % chall

    else:
        savedir = 'vae/vae_BCE_gradient_reducedCnnSeq-4layer_train-14_val-14'

    checkpointdir = os.path.join('./checkpoints', savedir)
    logdir = os.path.join('./logs', savedir)

    seed = random.randint(1, 100000)
    torch.manual_seed(seed)
    dataset_dir = '/media/gukyeong/HardDisk/dataset/CURE-TSR/folds/RealChallengeFree'

    if args.write_enable:
        os.mkdir(checkpointdir)
        writer = SummaryWriter(log_dir=logdir)
        print('log directory: %s' % logdir)
        print('checkpoints directory: %s' % checkpointdir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if dis_train:
        best_score = 1e20  # Initinalization of a variable for accuracy
        batch_size = 64  # there will be additional 64 fake samples
        d = models.DisShallowLinear(28 * 28 * 3)
        d = torch.nn.DataParallel(d).to(device)
        d.apply(models.weights_init)
        optimizer = optim.Adam(d.parameters(), lr=1e-3)

        if use_vae:
            vae = models.VAECURELinear()
            vae = torch.nn.DataParallel(vae).to(device)
            vae.eval()
            if os.path.isfile(vae_ckpt):
                print("=> loading checkpoint '{}'".format(vae_ckpt))
                checkpoint = torch.load(vae_ckpt)
                best_loss = checkpoint['best_loss']
                vae.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                      .format(vae_ckpt, checkpoint['epoch'], best_loss))
            else:
                print("=> no checkpoint found at '{}'".format(vae_ckpt))

    else:
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
                # best_loss = checkpoint['best_loss']
                vae.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
                      .format(vae_resume_ckpt, checkpoint['epoch'], checkpoint['best_loss']))
            else:
                print("=> no checkpoint found at '{}'".format(vae_resume_ckpt))

        # CURE-TSR (Train : Val : Test = 6 : 2 : 2)
        nsamples_fold = [325, 71, 17, 10, 235, 26, 295, 59, 20, 101, 16, 31, 179, 735]  # 14 classes
        in_cls = [13]
        out_cls = [i for i in range(13)]
        cls, nsamples_per_cls = utils.cal_nsample_perclass(in_cls, out_cls, nsamples_fold)

        in_train_loader = torch.utils.data.DataLoader(
            datasets.CURETSRdataset(os.path.join(dataset_dir, 'train'),
                                    transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                  transforms.ToTensor()]),
                                    target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                         transforms.ToTensor()]),
                                    cls=in_cls),
            batch_size=batch_size, shuffle=True)

        in_val_loader = torch.utils.data.DataLoader(
            datasets.CURETSRdataset(os.path.join(dataset_dir, 'val'),
                                    transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                  transforms.ToTensor()]),
                                    target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                         transforms.ToTensor()]),
                                    cls=in_cls),
            batch_size=batch_size, shuffle=True)

        out_train_loader = torch.utils.data.DataLoader(
            datasets.CURETSRdataset(os.path.join(dataset_dir[:-4], '%s' % chall, 'train'),
                                    transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                  transforms.ToTensor()]),
                                    target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                         transforms.ToTensor()]),
                                    cls=out_cls, nsamples_per_cls=[3 * i for i in nsamples_per_cls[len(in_cls)::]]),
            batch_size=batch_size, shuffle=True)

        out_val_loader = torch.utils.data.DataLoader(
            datasets.CURETSRdataset(os.path.join(dataset_dir[:-4], '%s' % chall, 'val'),
                                    transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                  transforms.ToTensor()]),
                                    target_transform=transforms.Compose([transforms.Resize([28, 28]),
                                                                         transforms.ToTensor()]),
                                    cls=out_cls, nsamples_per_cls=[i for i in nsamples_per_cls[len(in_cls)::]]),
            batch_size=batch_size, shuffle=True)

        # Start training
        timestart = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if dis_train:
                print('\n*** Start Training *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs))

                if use_vae:
                    d_train.train(d, device, in_train_loader, optimizer, epoch + 1, args.print_freq, vae=vae,
                                  std=std, out_iter=iter(out_train_loader))
                else:
                    d_train.train(d, device, in_train_loader, optimizer, epoch + 1, args.print_freq,
                                  out_iter=iter(out_train_loader))

                print('\n*** Start Testing *** Epoch: [%d/%d]\n' % (epoch + 1, args.epochs))

                if use_vae:
                    loss, acc, _ = d_train.test(d, device, in_val_loader, epoch + 1, args.print_freq, vae=vae,
                                                std=std, out_iter=iter(out_val_loader))
                else:
                    loss, acc, _ = d_train.test(d, device, in_val_loader, epoch + 1, args.print_freq,
                                                out_iter=iter(out_val_loader))

                is_best = loss < best_score
                best_score = min(loss, best_score)

            else:
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
                if dis_train:
                    if epoch % args.write_freq == 0 or is_best is True:
                        writer.add_scalar('loss', loss, epoch + 1)
                        writer.add_scalar('accuracy', acc, epoch + 1)
                        utils.save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': d.state_dict(),
                            'best_loss': best_score,
                            'last_loss': loss,
                            'optimizer': optimizer.state_dict(),
                        }, is_best, checkpointdir)

                else:
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
