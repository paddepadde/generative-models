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

    def __init__(self, beta=1.0):
        super(VAELoss, self).__init__()
        self.beta = beta
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x, x_hat, mu, logvar):
        # Loss of VAE is composed of reconstruction error (MSE between true input 
        # and reconstruction) and latent loss (KL divergence between learned latent 
        # distribution N(mu, sigma) and assumed true latent distribution N(0, 1))
        mse = self.mse(x_hat, x)
        d_kl = -.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar))
        return mse + self.beta * d_kl

class ConvVAE(nn.Module):

    def __init__(self, latent_size=10):
        super(ConvVAE, self).__init__()
        
        # Layers for encoder network X -> z
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1))

        self.encoder_fc_1 = nn.Linear(16 * 10 * 10, 512)
        self.encoder_mu = nn.Linear(512, latent_size)        
        self.encoder_logvar = nn.Linear(512, latent_size)     

        # Layers for decoder network z -> \hat{X}
        self.decoder_fc_1 = nn.Linear(latent_size, 512)
        self.decoder_fc_2 = nn.Linear(512, 16 * 10 * 10)

        self.deconv1 = nn.ConvTranspose2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(2, 2), output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1, kernel_size=(3, 3),  stride=(1, 1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_hat = self.decode(z)
        return x, x_hat, mu, logvar, z

    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 16 * 10 * 10)
        x = F.relu(self.encoder_fc_1(x))

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
        z = F.relu(self.decoder_fc_1(z))
        z = F.relu(self.decoder_fc_2(z))

        z = z.view(-1, 16, 10, 10)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        z = F.relu(self.deconv3(z))
        x_hat = z.view(-1, 1, 28, 28)

        return x_hat