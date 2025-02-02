import numpy as np 
import torch.nn as nn

# custom initialization for generator and discriminator networks
# sample weights from normal distrib with mu=0, sigma=0.02
# see: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1: 
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)