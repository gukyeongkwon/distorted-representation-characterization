import torch
import torch.utils.data
import torchvision.transforms as transforms

import os
import numpy as np

import models
import datasets
import vae_train


class VanillaBackprop:
    """
    Produces gradients generated with vanilla back propagation from the image
    """
    def __init__(self, model, loss_type=None):
        self.model = model
        self.loss_type = loss_type
        self.gradients = {}
        self.z = None
        self.model.eval()
        self.hook_layers()

    def save_grad(self, key, is_act=False):
        """
        Args:
            key (str): A key for the saved gradeints (self.gradients)
            is_act (bool): If true, hook the activation to obtian the gradients for the ouput of layers
        Returns:
            hook_function or hook_function_act
        """
        def hook_function(module, grad_in, grad_out):
            """
            Args:
                grad_in (tuple): grad_in[0]: input, grad_in[1]: wegights, grad_in[2]: bias
                                (if the layer is a Linear layer, grad_in[2] corresponds to the weight gradient)
            """
            self.gradients[key] = grad_in[1]

        def hook_function_act(grad):
            self.gradients[key] = grad

        if is_act is False:
            return hook_function
        else:
            return hook_function_act

    def hook_layers(self):
        # Register hook to each conv layers
        down_stream = list(self.model.module.down._modules.items())
        for (idx, layer) in down_stream:
            if int(idx) % 2 == 1:
                continue
            else:
                layer.register_backward_hook(self.save_grad('down_' + idx))

        self.model.module.fc11.register_backward_hook(self.save_grad('fc11'))

        up_stream = list(self.model.module.up._modules.items())
        for (idx, layer) in up_stream:
            if int(idx) % 2 == 1:
                continue
            else:
                layer.register_backward_hook(self.save_grad('up_' + idx))

    def generate_gradients(self, input_image, target_image):
        # Zero grads
        self.model.zero_grad()

        input_image.requires_grad = True
        input_image.register_hook(self.save_grad('input', is_act=True))
        # Used activation hook because of  the Linear layer exception described in the comments of hook_function
        self.model.module.fc11.weight.register_hook(self.save_grad('fc11', is_act=True))
        mu, logvar = self.model.module.encode(input_image)
        self.z = self.model.module.reparameterize(mu, logvar)
        self.z.register_hook(self.save_grad('latent', is_act=True))
        recon = self.model.module.decode(self.z)

        loss = vae_train.bernoulli_loss_function(recon, target_image, mu, logvar, beta=1,
                                                 imgsize=recon.view(input_image.size(0), -1).size(1),
                                                 loss_type=self.loss_type)
        loss.backward()
        return


def main():
    dataset_dir = '/home/gukyeong/dataset/mnist'
    vae_ckpt = './checkpoints/mnist/vae/gradient/vae_BCE_gradient_reducedCnnSeq-4layer_in-0-5_out-6-9/' \
               'model_best.pth.tar'

    vae = models.VAEReducedCNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vae = torch.nn.DataParallel(vae).to(device)
    if os.path.isfile(vae_ckpt):
        print("=> loading checkpoint '{}'".format(vae_ckpt))
        checkpoint = torch.load(vae_ckpt)
        best_loss = checkpoint['best_loss']
        vae.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}, best_loss {})"
              .format(vae_ckpt, checkpoint['epoch'], best_loss))
    else:
        print("=> no checkpoint found at '{}'".format(vae_ckpt))

    vbackprop = VanillaBackprop(vae)

    nsamples_perclass = 10
    test_nsamples_perclass = [nsamples_perclass] * 10

    ##########################################################################################
    # Gradients Stat.
    grad_stat = np.zeros([sum(test_nsamples_perclass), 4])
    label_cnt = [0] * 10

    # Vanilla Backpropagation Visualization
    # class_cnt = [0] * 10

    # Gradient Visualization on Latent Space
    # z = np.zeros([sum(test_nsamples_perclass), 2])
    # gradients = np.zeros([sum(test_nsamples_perclass), 2])
    # labels = np.zeros([sum(test_nsamples_perclass),])
    batch_size = 16
    test_loader = torch.utils.data.DataLoader(
        datasets.SubsetMNISTdataset(dataset_dir, train=False, transform=transforms.ToTensor(),
                                    target_transform=transforms.ToTensor(),
                                    nsamples_perclass=test_nsamples_perclass),
        batch_size=batch_size, shuffle=True)

    for batch_idx, (input_img, target_img, label) in enumerate(test_loader):
        print('Processing (%d / %d)...' % (batch_idx, len(test_loader)))
        input_img = input_img.to(device)
        input_img.requires_grad = True
        target_img = target_img.to(device)
        vbackprop.generate_gradients(input_img, target_img)


if __name__ == '__main__':
    main()
