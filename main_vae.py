import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from models import VAE, VAELoss, ConvVAE

# Load and store the MNIST train data 
transform = transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)

vae = ConvVAE(latent_size=10)
criterion = VAELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

z_test = np.random.normal(size=(9, 6))
z_test = torch.from_numpy(z_test).float()

epochs = 5
for e in range(epochs):
    running_loss = 0
    for (idx, data) in enumerate(trainloader):
        x, _ = data

        optimizer.zero_grad()

        x, x_hat, mu, logvar, _ = vae(x)
        loss = criterion(x, x_hat, mu, logvar)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print(loss.item())

        if idx % 50 == 49:
            # setup z for decoder network input.
            z_test = np.random.normal(size=(60, 10))
            z_test = torch.from_numpy(z_test).float()

            x_hat = vae.decode(z_test)
            # reshape into old image format 
            x_hat = x_hat.view((-1, 1, 28, 28))

            # create and visualize grid
            grid = torchvision.utils.make_grid(x_hat, nrow=10, padding=5)
            plt.figure(figsize=(15, 6))
            plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
            plt.show()

        

