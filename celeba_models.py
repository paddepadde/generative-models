import torch 
import torch.nn as nn
import torch.nn.functional as F

class ConvVAE(nn.Module):

    def __init__(self, latent_size=10):
        super(ConvVAE, self).__init__()

        self.latent_size = latent_size

        ## ENCODER
        # image size -> (64 - 4) / 2 + 1 -> 31
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(4, 4), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)

        # image size -> (31 - 4) / 2 + 1 -> 14.5 -> 14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.batch_norm2 = nn.BatchNorm2d(64)

        # image size -> (14 - 3) + 1 -> 12
        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1))

        self.encoder_fc1 = nn.Linear(32 * 12 * 12, 512)
        self.encoder_mu = nn.Linear(512, latent_size)
        self.encoder_logvar = nn.Linear(512, latent_size)

        ## DEOCDER
        self.decoder_fc1 = nn.Linear(latent_size, 512)
        self.decoder_fc2 = nn.Linear(512, 32 * 12 * 12)

        self.deconv1 = nn.ConvTranspose2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=(2, 2), output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=(4, 4), stride=(2, 2))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_hat = self.decode(z)

        return x, x_hat, mu, logvar, z

    def encode(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = F.relu(self.conv3(x))

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.encoder_fc1(x))
        
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)

        return mu, logvar

    def sample(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.empty_like(sigma).normal_()

        z = mu + eps * sigma
        return z

    def decode(self, z):
        z = F.relu(self.decoder_fc1(z))
        z = F.relu(self.decoder_fc2(z))

        z = z.view(-1, 32, 12, 12)
        z = F.relu(self.deconv1(z))
        z = F.relu(self.deconv2(z))
        x_hat = F.relu(self.deconv3(z))

        return x_hat
        
class Discriminator5(nn.Module):

    def __init__(self):
        super(Discriminator5, self).__init__()

        # self.dropout = nn.Dropout2d(0.2)

        # image size -> (64 - 4) / 2 + 1 -> 31
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(4, 4), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(256)
       
        # image size -> (31 - 5) / 2 + 1 -> 14
        self.conv2 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2))
        self.batch_norm2 = nn.BatchNorm2d(512)

        # image size -> (14 - 4) + 1 -> 11
        self.conv3 = nn.Conv2d(512, 128, kernel_size=(4, 4), stride=(1, 1))
        self.batch_norm3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 11 * 11, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.batch_norm2(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.batch_norm3(x)

        x = torch.flatten(x, start_dim=1)

        x = self.sigmoid(self.fc1(x))
        return x

class Generator4(nn.Module):

    def __init__(self, latent_size=100):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(latent_size, 512)
        self.fc2 = nn.Linear(512, 128 * 11 * 11)
        
        self.deconv1 = nn.ConvTranspose2d(128, 512, kernel_size=(4, 4), stride=(1, 1))
        self.batch_norm1 = nn.BatchNorm2d(512)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), output_padding=1)
        self.batch_norm2 = nn.BatchNorm2d(256)

        self.deconv3 = nn.ConvTranspose2d(256, 1, kernel_size=(4, 4), stride=(2, 2))
        self.tanh = nn.Tanh()
        
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        x = x.view(-1, 128, 11, 11)

        x = F.relu(self.batch_norm1(self.deconv1(x)))
        x = F.relu(self.batch_norm2(self.deconv2(x)))
        x = self.deconv3(x)

        x = self.tanh(x)

        return x

# Initial network structure based on the models used in this tutorial:
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        # input latent space (?, 100)
        # rescale to (?, 100, 1, 1) first
        self.deconv1 = nn.ConvTranspose2d(100, 512, kernel_size=(4, 4), stride=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(512)

        # image size -> (?, 512, 4, 4)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(256)

        # image size -> (?, 256, 8, 8)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.batchnorm3 = nn.BatchNorm2d(128)

        # image size -> (?, 128, 16, 16)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.batchnorm4 = nn.BatchNorm2d(64)

        # image size -> (?, 64, 32, 32)
        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        
        # image size -> (?, 3, 64, 64)
        # TODO: change to functional
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = F.relu(self.batchnorm1(self.deconv1(x)))
        x = F.relu(self.batchnorm2(self.deconv2(x)))
        x = F.relu(self.batchnorm3(self.deconv3(x)))
        x = F.relu(self.batchnorm4(self.deconv4(x)))
        x = self.tanh(self.deconv5(x))
        return x

class Generator2(nn.Module):
    def __init__(self, ngpu=1, latent_size=100):
        super(Generator2, self).__init__()
        self.ngpu = 1
        ngf = 64
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(100, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator2(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator2, self).__init__()
        self.ngpu = ngpu
        ndf = 64
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        # TODO: test with dropout
        self.dropout = nn.Dropout2d(0.2)

        # input is image batch (?, 3, 64, 64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(64)

        # (64 - 4 + 2) / 2 + 1 -> 32
        # image size -> (?, 64, 32, 32)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(128)

        # (32 - 4 + 2) / 2 + 1 -> 16
        # image size -> (?, 128, 16, 16)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.batchnorm3 = nn.BatchNorm2d(256)

        # (16 - 4 + 2) / 2 + 1 -> 8
        # image size -> (?, 256, 8, 8)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        self.batchnorm4 = nn.BatchNorm2d(512)

        # (8 - 4 + 2) / 2 + 1  -> 4
        # image size -> (?, 512, 4, 4)
        self.conv5 = nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1))

        # (4 -4) + 1  -> 1
        # image size -> (?, 1, 1, 1) -> (?, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.batchnorm3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.batchnorm4(self.conv4(x)), 0.2)
        x = self.conv5(x)
        x = self.sigmoid(x)
        x = x.view(-1, 1)

        return x
