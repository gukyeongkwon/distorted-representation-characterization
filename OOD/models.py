import torch
import torch.nn as nn


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