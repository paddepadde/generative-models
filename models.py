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

class Generator2(nn.Module):
    def __init__(self, noise_size):
        super(Generator2, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(noise_size, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Generator(nn.Module):

    def __init__(self, noise_size=100):
        super(Generator, self).__init__()
        
        self.init_size = 32 // 4
        self.fc_1 = nn.Linear(noise_size, 128 * self.init_size ** 2)
        
        self.batch_norm0 = nn.BatchNorm2d(128)
        self.upsample0 = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(128, 64, 3, 1, 1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 1, 3, 1, 1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.fc_1(x)
        x = x.view(x.shape[0], 128, self.init_size, self.init_size)

        x = self.batch_norm0(x)
        x = self.upsample0(x)

        x = self.leaky_relu(self.batch_norm1(self.conv1(x)))
        x = self.upsample0(x)

        x = self.leaky_relu(self.batch_norm2(self.conv2(x)))
        x = self.tanh(self.conv3(x))
        return x

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
       
        self.conv1 = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.25)

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.downsampled_size = 32 // 2 ** 4
        self.fc1 = nn.Linear(128 * self.downsampled_size ** 2, 1)
        self.sogmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.batch_norm2(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.batch_norm3(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.batch_norm4(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.sogmoid(x)
        return x


class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
