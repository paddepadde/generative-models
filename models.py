import torch
import torch.nn as nn 
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, input_size, hidden_size=64, latent_size=6):
        super(VAE, self).__init__()

        self.input_size = input_size

        # FC layers for encoder network X -> z
        self.encoder_1 = nn.Linear(input_size, hidden_size)
        self.encoder_2 = nn.Linear(hidden_size, hidden_size)
        self.encoder_mu = nn.Linear(hidden_size, latent_size)
        self.encoder_logvar = nn.Linear(hidden_size, latent_size)

        # FC layers for decoder network z -> \hat{X}
        self.decoder_1 = nn.Linear(latent_size, hidden_size)
        self.decoder_2 = nn.Linear(hidden_size, hidden_size)
        self.decoder_3 = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.view(-1, self.input_size) # reshape into shape (?, 28 * 28)
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_hat = self.decode(z)
        return x, x_hat, mu, logvar, z

    def encode(self, x):
        x = F.relu(self.encoder_1(x))
        x = F.relu(self.encoder_2(x))
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)

        return mu, logvar

    def sample(self, mu, logvar):
        # create epsilon with same size as sigma
        sigma = torch.exp(0.5 * logvar)
        eps = torch.empty_like(sigma).normal_()
        # use reparametrization trick (Kingma, 2013) to compute z
        z = mu + eps * sigma
        return z

    def decode(self, z):
        # TODO: maybe make bernouli distribution for correct image pixel values in (0, 1)
        z = F.relu(self.decoder_1(z))
        z = F.relu(self.decoder_2(z))
        x = self.decoder_3(z)
        return x

class VAELoss(nn.Module):

    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x, x_hat, mu, logvar):
        # Loss of VAE is composed of reconstruction error (MSE between true input 
        # and reconstruction) and latent loss (KL divergence between learned latent 
        # distribution N(mu, sigma) and assumed true latent distribution N(0, 1))
        mse = self.mse(x_hat, x)
        d_kl = -.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        return mse + d_kl


