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
    def __init__(self, transforms_=None, mode="train"):
#         self.transform = transforms.Compose(transforms_)

        self.files = sorted([path[0:-4] for path in glob.glob('/home/matthewachan/datasets/FSS-1000/fewshot_data/**/*.jpg',recursive=True)])
#         if mode == "train":
#             self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        
        

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]+".jpg")
        mask = Image.open(self.files[index % len(self.files)]+".png")
        messymask = Image.fromarray(np.uint8(self.noisy_blur(np.array(mask))[:,:,0:3]))
                
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
            
#         # Random rotation
#         rotate = transforms.RandomRotation(45)
#         img = rotate(img)
#         mask = rotate(mask)
#         messymask = rotate(messymask)
            
        img = np.array(img)
        mask = np.array(mask)
        messymask = np.array(messymask)
        
        img = np.concatenate((img,messymask[:,:,0:1]), axis = 2)
        
        gtmask = torch.from_numpy(mask[:,:,0:1])
        imgin = torch.from_numpy(img)
        
        gtmask = gtmask.permute(2,0,1)
        imgin = imgin.permute(2,0,1)

        return {"gtmask": gtmask, "imgin": imgin, "path": self.files[index]}

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