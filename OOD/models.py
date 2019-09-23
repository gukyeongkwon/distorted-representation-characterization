import torch
import torch.nn as nn


class VAELinear(nn.Module):
    def __init__(self):
        super(VAELinear, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 2)
        self.fc22 = nn.Linear(400, 2)
        self.fc3 = nn.Linear(2, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h2 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h2))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        return out, mu, logvar


class VAECNN(nn.Module):
    def __init__(self):
        super(VAECNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=2)

        self.conv2_bn = nn.BatchNorm2d(32, track_running_stats=True)
        self.conv3_bn = nn.BatchNorm2d(64, track_running_stats=True)

        self.fc11 = nn.Linear(3 * 3 * 64, 20)
        self.fc12 = nn.Linear(3 * 3 * 64, 20)
        self.fc2 = nn.Linear(20, 3 * 3 * 64)

        self.dconv1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=2)  # output 4x4
        self.dconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # output 8x8
        self.dconv3 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=2)  # output 14x14
        self.dconv4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)  # output 28x28

        self.dconv2_bn = nn.BatchNorm2d(32, track_running_stats=True)
        self.dconv3_bn = nn.BatchNorm2d(32, track_running_stats=True)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2_bn(self.conv2(h1)))
        h3 = self.relu(self.conv3_bn(self.conv3(h2)))
        h4 = self.relu(self.conv4(h3))
        return self.fc11(h4.view(-1, 3 * 3 * 64)), self.fc12(h4.view(-1, 3 * 3 * 64))


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4_d = self.fc2(z)
        h3_d = self.relu(self.dconv1(h4_d.view(-1, 64, 3, 3)))
        h2_d = self.relu(self.dconv2_bn(self.dconv2(h3_d)))
        h1_d = self.relu(self.dconv3_bn(self.dconv3(h2_d)))
        return self.sigmoid(self.dconv4(h1_d))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAEReducedCNN(nn.Module):
    def __init__(self):
        super(VAEReducedCNN, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(1, 2, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(2, 4, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 12, 4, stride=2, padding=2),
            nn.ReLU()
        )

        self.fc11 = nn.Linear(3 * 3 * 12, 20)
        self.fc12 = nn.Linear(3 * 3 * 12, 20)
        self.fc2 = nn.Linear(20, 3 * 3 * 12)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(12, 8, 4, stride=2, padding=2),  # output 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1),  # output 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, 4, stride=2, padding=2),  # output 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, 4, stride=2, padding=1),  # output 28x28
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h4 = self.down(x)
        return self.fc11(h4.view(-1, 3 * 3 * 12)), self.fc12(h4.view(-1, 3 * 3 * 12))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4_d = self.fc2(z)
        recon = self.up(h4_d.view(-1, 12, 3, 3))
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAECNNIndLatent(nn.Module):
    def __init__(self):
        super(VAECNNIndLatent, self).__init__()

        self.conv1 = nn.Conv2d(1, 2, 4, stride=2, padding=2)
        self.conv2 = nn.Conv2d(2, 4, 4, stride=2, padding=2)
        self.conv3 = nn.Conv2d(4, 8, 4, stride=2, padding=2)
        self.conv4 = nn.Conv2d(8, 12, 4, stride=2, padding=2)

        self.conv2_bn = nn.BatchNorm2d(4, track_running_stats=True)
        self.conv3_bn = nn.BatchNorm2d(8, track_running_stats=True)

        self.fc11 = nn.Linear(3 * 3 * 12, 2)
        self.fc12 = nn.Linear(3 * 3 * 12, 2)
        self.fc2 = nn.Linear(2 * 9, 3 * 3 * 12)

        self.dconv1 = nn.ConvTranspose2d(12, 8, 4, stride=2, padding=2)  # output 4x4
        self.dconv2 = nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1)  # output 8x8
        self.dconv3 = nn.ConvTranspose2d(4, 2, 4, stride=2, padding=2)  # output 14x14
        self.dconv4 = nn.ConvTranspose2d(2, 1, 4, stride=2, padding=1)  # output 28x28

        self.dconv2_bn = nn.BatchNorm2d(4, track_running_stats=True)
        self.dconv3_bn = nn.BatchNorm2d(2, track_running_stats=True)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.class_of_interest = 5

    def encode(self, x):
        h1 = self.relu(self.conv1(x))
        h2 = self.relu(self.conv2_bn(self.conv2(h1)))
        h3 = self.relu(self.conv3_bn(self.conv3(h2)))
        h4 = self.relu(self.conv4(h3))
        return self.fc11(h4.view(-1, 3 * 3 * 12)), self.fc12(h4.view(-1, 3 * 3 * 12))

    def reparameterize(self, mu, logvar, label):
        z = torch.zeros([mu.shape[0], 2 * 9]).to('cuda')

        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            tmp_z = eps.mul(std).add_(mu)
        else:
            tmp_z = mu

        for i in range(mu.shape[0]):
            if label[i] > self.class_of_interest:
                z[i, (label[i] - 1) * 2 : label[i] * 2] = tmp_z[i, :]
            else:
                z[i, label[i] * 2 : (label[i] + 1) * 2] = tmp_z[i, :]

        return z

    def decode(self, z):
        h4_d = self.fc2(z)
        h3_d = self.relu(self.dconv1(h4_d.view(-1, 12, 3, 3)))
        h2_d = self.relu(self.dconv2_bn(self.dconv2(h3_d)))
        h1_d = self.relu(self.dconv3_bn(self.dconv3(h2_d)))
        return self.sigmoid(self.dconv4(h1_d))

    def forward(self, x, label):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, label)
        return self.decode(z), mu, logvar


class VAECURECNN(nn.Module):
    def __init__(self):
        super(VAECURECNN, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(3, 3, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(3, 6, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(6, 9, 4, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(9, 12, 4, stride=2, padding=2),
            nn.ReLU()
        )

        self.fc11 = nn.Linear(3 * 3 * 12, 20)
        self.fc12 = nn.Linear(3 * 3 * 12, 20)
        self.fc2 = nn.Linear(20, 3 * 3 * 12)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(12, 9, 4, stride=2, padding=2),  # output 4x4
            nn.ReLU(),
            nn.ConvTranspose2d(9, 6, 4, stride=2, padding=1),  # output 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(6, 3, 4, stride=2, padding=2),  # output 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),  # output 28x28
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h4 = self.down(x)
        return self.fc11(h4.view(-1, 3 * 3 * 12)), self.fc12(h4.view(-1, 3 * 3 * 12))

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h4_d = self.fc2(z)
        recon = self.up(h4_d.view(-1, 12, 3, 3))
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAECURELinear(nn.Module):
    def __init__(self):
        super(VAECURELinear, self).__init__()

        self.down = nn.Sequential(
            nn.Linear(28 * 28 * 3, 1000),
            nn.ReLU(),
            nn.Linear(1000, 400),
            nn.ReLU(),
        )

        self.fc11 = nn.Linear(400, 20)
        self.fc12 = nn.Linear(400, 20)
        self.fc2 = nn.Linear(20, 400)

        self.up = nn.Sequential(
            nn.Linear(400, 1000),
            nn.ReLU(),
            nn.Linear(1000, 28 * 28 * 3),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.down(x)
        return self.fc11(h), self.fc12(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h_d = self.fc2(z)
        recon = self.up(h_d.view(-1, 12, 3, 3))
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAECUREGrad(nn.Module):
    def __init__(self):
        super(VAECUREGrad, self).__init__()

        self.down = nn.Sequential(
            nn.Conv2d(3, 3, 4, stride=2, padding=2, bias=False),
            nn.Threshold(0, 1),
            nn.Conv2d(3, 6, 4, stride=2, padding=2, bias=False),
            nn.Threshold(0, 1),
            nn.Conv2d(6, 9, 4, stride=2, padding=2, bias=False),
            nn.Threshold(0, 1),
            nn.Conv2d(9, 12, 4, stride=2, padding=2, bias=False),
            # nn.Threshold(0, 1),
        )

        self.fc11 = nn.Linear(3 * 3 * 12, 20, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h4 = self.sigmoid(x) * (1 - self.sigmoid(x))
        h1 = self.down(h4)
        return self.fc11(h1.view
                         (-1, 3 * 3 * 12))


class DisShallowLinear(nn.Module):
    def __init__(self, dim):
        super(DisShallowLinear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.classifier(x)
        return y


class DisLinear(nn.Module):
    def __init__(self):
        super(DisLinear, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 20)
        self.fc4 = nn.Linear(20, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x.view(-1, 28*28)))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


class DisCNN(nn.Module):
    def __init__(self):
        super(DisCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 128, 5, stride=2, padding=2)

        self.conv2_bn = nn.BatchNorm2d(32, track_running_stats=True)
        self.conv3_bn = nn.BatchNorm2d(64, track_running_stats=True)

        self.fc = nn.Linear(2*2*128, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        conv_out = self.relu(self.conv3_bn(self.conv3(x)))
        x = self.conv4(conv_out)
        x = self.sigmoid(self.fc(x.view(-1, 2*2*128)))
        return x, conv_out


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.max = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        conv_out = self.features(x)
        max_out = self.max(conv_out)
        x = max_out.view(max_out.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x, conv_out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)