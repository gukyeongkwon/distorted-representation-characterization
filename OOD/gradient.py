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
        # Used activation hook because of the Linear layer exception described in the comments of hook_function
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