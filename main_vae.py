import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt

from models import VAE, VAELoss

# Load and store the MNIST train data 
transform = transforms.Compose([
    torchvision.transforms.ToTensor(), 
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200, shuffle=True)

vae = VAE(input_size=28*28)
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

        x, x_hat, mu, logvar = vae(x)
        loss = criterion(x, x_hat, mu, logvar)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if idx % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (e + 1, idx + 1, running_loss / 200))
            running_loss = 0.0

    x_hat = vae.decode(z_test)
    x_hat = x_hat.detach().numpy()
    x_hat = np.reshape(x_hat, (9, 1, 28, 28))
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_hat[i], cmap='gray')
    plt.show()
        

