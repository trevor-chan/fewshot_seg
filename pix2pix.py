import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=4, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=1000, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between model checkpoints")
opt = parser.parse_args()
#print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# Loss functions
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss(reduction='mean')

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 5

# Regularization constants for discriminator gradient penalty
r1_lambda = 5
r1_k = 16

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
generator = GeneratorUNet()
discriminator = Discriminator()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()

if opt.epoch != 0:
    # Load pretrained models
    generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
# transforms_ = [
#     transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(degrees=(-45, 45)),
#     transforms.RandomAutocontrast(),
# ]

dataloader = DataLoader(
#     ImageDataset(transforms_=transforms_),
    ImageDataset(),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8,
)

val_dataloader = DataLoader(
#     ImageDataset(transforms_=transforms_, mode="val"),
    ImageDataset(mode="val"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(val_dataloader))
    gtmask = Variable(imgs["gtmask"].type(Tensor))
    imgin = Variable(imgs["imgin"].type(Tensor))
    genmask = generator(imgin)
    
    img_sample = torch.cat((imgin.data[:,0:3,:,:], 
                            imgin.data[:,3:4,:,:].repeat(1,3,1,1)*255, 
                            gtmask.data.repeat(1,3,1,1), 
                            genmask.data.repeat(1,3,1,1)*255), 
                           dim=2)

#     img_sample = torch.cat((imgin.data[:,3:4,:,:]*255, 
#                             gtmask.data, 
#                             genmask.data*255, ),
#                             dim=2)
    
    save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True, normalize_each=True)


# ----------
#  Training
# ----------

prev_time = time.time()

pixloss = []
genloss = []
disloss = []

for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Model inputs
        gtmask = Variable(batch["gtmask"].type(Tensor))
        imgin = Variable(batch["imgin"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((gtmask.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((gtmask.size(0), *patch))), requires_grad=False)
        
        
#         print(valid.shape)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # GAN loss
        genmask = generator(imgin)
        
        pred_fake = discriminator(imgin, genmask)
        
        loss_GAN = criterion_GAN(pred_fake, valid)
        # Pixel-wise loss
        loss_pixel = criterion_pixelwise(gtmask, genmask)

        # Total loss
        loss_G = loss_GAN + lambda_pixel * loss_pixel
#         loss_G = lambda_pixel * loss_pixel
        
        loss_G.backward()

        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Real loss
        gtmask.requires_grad = True
        pred_real = discriminator(imgin, gtmask)
        loss_real = criterion_GAN(pred_real, valid)

        # Fake loss
        pred_fake = discriminator(imgin, genmask.detach())
        loss_fake = criterion_GAN(pred_fake, fake)
        
#         print(loss_real)
#         print('------')
#         print(loss_fake)

        # Regularization
        if i%r1_k == 0:
            grad_real, = torch.autograd.grad(
                outputs=pred_real.sum(), inputs=gtmask, create_graph=True, allow_unused=True
            )
            grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
            grad_penalty = (r1_lambda / 2 ) * grad_penalty * r1_k
        else:
            grad_penalty = 0
            
        # Total loss
        loss_D = 0.5 * (loss_real + loss_fake) + grad_penalty

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f, grad_penalty: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
            % (
                epoch,
                opt.n_epochs,
                i,
                len(dataloader),
                loss_D.item(),
                grad_penalty,
                loss_G.item(),
                loss_pixel.item()*lambda_pixel,
                loss_GAN.item(),
                time_left,
            )
        )
        
        pixloss.append(loss_pixel.item())
        genloss.append(loss_GAN.item())
        disloss.append(loss_D.item())

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)
            if batches_done == 0: continue
            #also plot losses
            x = range(epoch*1250+i+1)
            #plt.plot(x,pixloss, label='pixel loss')
            plt.plot(x,genloss, label='generator loss')
            plt.plot(x,disloss, label='discriminator loss')
            plt.savefig('saved_models/losses.png',format='png')


    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))