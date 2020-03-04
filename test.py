# # Import disthe required libraries
import torch
import torch.nn as nn 
import torchvision
import numpy as np 
import matplotlib.pyplot as plt
from celeba_models import Generator, Discriminator
from util import weight_init

# Hyperparameters for the training procedure
batch_size = 200
device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

# Transformations applied to the initial images
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(64),
    torchvision.transforms.CenterCrop(64),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the images from the `data/CELEBA` folder
dataset = torchvision.datasets.ImageFolder('./data/CELEBA', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


latent_size = 100 # size of z vector
generator = Generator().to(device)
generator.apply(weight_init)
discriminator = Discriminator().to(device)
discriminator.apply(weight_init)

criterion = nn.BCELoss()
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

# Save loss and accuracy for plot
list_discriminator_fake = []
list_discriminator_real = []
list_generator = []

epochs = 2
label_smooth = 0.0
real_labels, fake_labels = 1.0 - label_smooth, 0.0

y_real = torch.full((batch_size, 1), real_labels).to(device)
y_fake = torch.full((batch_size, 1), fake_labels).to(device)

discriminator_steps = 2
generator_steps = 1

for e in range(epochs):
    for (idx, data) in enumerate(dataloader, 0):

        running_discriminator_loss = 0
    
        for _ in range(discriminator_steps):
            discriminator_optimizer.zero_grad()

            # Train discriminator on real training data
            x, _ = data
            x = x.to(device)

            y_pred = discriminator(x)
            error_real = criterion(y_pred, y_real)
            error_real.backward()

            # Train discriminator on fake training data
            noise = torch.rand((batch_size, latent_size)).to(device)
            x_fake = generator(noise)
            y_pred = discriminator(x_fake.detach())
            error_fake = criterion(y_pred, y_fake)
            error_fake.backward()

            discriminator_optimizer.step()

            running_discriminator_loss += error_real.item() + error_fake.item()
       
        # TODO: consider all iterations
        list_discriminator_fake.append(error_fake.item())
        list_discriminator_real.append(error_real.item())

        running_generator_loss = 0
        for _ in range(generator_steps):

            # Train generator to generate fake data
            generator_optimizer.zero_grad()
            noise = torch.rand((batch_size, latent_size)).to(device)
            x_fake = generator(noise)
            y_pred = discriminator(x_fake)
            generator_loss = criterion(y_pred, y_real)
            generator_loss.backward()

            generator_optimizer.step()

            running_generator_loss += generator_loss.item()      

        list_generator.append(running_generator_loss)

        if idx % 10 == 0:
            print("Epoch {}, Iteration {}/{} \t Loss Gen. {:.2f}, Loss Dic. {:.2f}".format(
                e, idx+1, len(dataset) // batch_size, 
                running_generator_loss, running_discriminator_loss
            ))

            noise = torch.rand((32, latent_size)).to(device)
            x = generator(noise)

            plt.figure(figsize=(15, 8))
            grid = torchvision.utils.make_grid(x, nrow=8, padding=4)
            plt.imshow(grid.permute(1, 2, 0).detach().cpu().numpy())
            plt.title('Generated Faces')
            plt.axis('off')
            plt.show()