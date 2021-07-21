import glob
import random
import os
import numpy as np
from skimage import filters
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, transforms_=None, mode="train", dirpath = None):
#         self.transform = transforms.Compose(transforms_)
        self.mode = mode
        if dirpath == None: dirpath = '/home/matthewachan/datasets/FSS-1000/fewshot_data'
        self.files = sorted([path[0:-4] for path in glob.glob(dirpath+'/**/*.jpg',recursive=True)])

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]+".jpg")
        mask = Image.open(self.files[index % len(self.files)]+".png")
        
        if self.mode == "train":
            messymask = Image.fromarray(np.uint8(self.random_noisy_blur(np.array(mask))[:,:,0:3]))
                        
            # Random rotation
            theta = np.random.normal(loc = 0.0, scale = 20, size = None)

            img = transforms.functional.rotate(img,theta)
            mask = transforms.functional.rotate(mask,theta)
            messymask = transforms.functional.rotate(messymask,theta)

            # Resize
            resize = transforms.Resize(size=(300, 300))
            img = resize(img)
            mask = resize(mask)
            messymask = resize(messymask)

            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(256, 256))
            img = transforms.functional.crop(img, i, j, h, w)
            mask = transforms.functional.crop(mask, i, j, h, w)
            messymask = transforms.functional.crop(messymask, i, j, h, w)

            # Random horizontal flipping
            if random.random() > 0.5:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
                messymask = transforms.functional.hflip(messymask)
                
            messymask = np.array(messymask)
            
        if self.mode == "test":
            # Resize
            resize = transforms.Resize(size=(256, 256))
            img = resize(img)
            mask = resize(mask)
                
        img = np.array(img)
        mask = np.array(mask)
        
        if self.mode == "train":
            img = np.concatenate((img,messymask[:,:,0:1]), axis = 2)
            gtmask = torch.from_numpy(mask[:,:,0:1])
            gtmask = gtmask.permute(2,0,1)
        else:
            img = np.concatenate((img,mask[:,:,0:1]),axis = 2)
        
        imgin = torch.from_numpy(img)
        imgin = imgin.permute(2,0,1)
        
        if self.mode == "train":
            return {"gtmask": gtmask, "imgin": imgin, "path": self.files[index]}
        else:
            return {"imgin": imgin, "path": self.files[index]}

    def __len__(self):
        return len(self.files)
    
    def reg_blur(self, label = None, threshold = 0.25):
        if label is None: return 1
        if type(label) is not np.ndarray: return 1
        blur = filters.gaussian(label,10,multichannel=True)
        return np.where(np.divide(blur,np.amax(blur))>threshold,1,0)
    
    def noisy_blur(self, label = None, threshold = 0.25, noise = .25, noise_kernels = 10):
        if label is None: 
            print("ERROR")
            return 1
        if type(label) is not np.ndarray: 
            print("ERROR")
            return 1
        blur = filters.gaussian(label,10,multichannel=True)
        
        noisetensor = torch.normal(0,1,size=(1,1,noise_kernels,noise_kernels))        
        noisetensor = torch.nn.functional.interpolate(noisetensor, size = blur.shape[:2], mode = 'bilinear', align_corners = False)
        noisetensor = np.array(noisetensor.permute(2,3,1,0)[:,:,:,0])
        noisetensor = np.divide(noisetensor,np.amax(np.array(noisetensor)))*np.amax(blur)*noise
        
        blur = blur.astype(np.float32)+noisetensor
        
        return np.where(np.divide(blur,np.amax(blur))>threshold,1,0)
    
    def random_noisy_blur(self, label = None, threshold = 0.25, noise = .3, noise_kernels = 10):
        if label is None: 
            print("ERROR")
            return 1
        if type(label) is not np.ndarray: 
            print("ERROR")
            return 1
        blur = filters.gaussian(label,10,multichannel=True)
        
        noise_kernels = noise_kernels+int((random.random()-.5)*6)
        noise = noise+(random.random()-.5)*.1
        threshold = threshold+(random.random()-.5)*.1
        
        noisetensor = torch.normal(0,1,size=(1,1,noise_kernels,noise_kernels))        
        noisetensor = torch.nn.functional.interpolate(noisetensor, size = blur.shape[:2], mode = 'bilinear', align_corners = False)
        noisetensor = np.array(noisetensor.permute(2,3,1,0)[:,:,:,0])
        noisetensor = np.divide(noisetensor,np.amax(np.array(noisetensor)))*np.amax(blur)*noise
        
        blur = blur.astype(np.float32)+noisetensor
        
        return np.where(np.divide(blur,np.amax(blur))>threshold,1,0)