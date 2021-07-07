import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.files = sorted(glob.glob('datasets/FSS-1000/fewshot_data/*/*.jpg')[0:-4])
#         if mode == "train":
#             self.files.extend(sorted(glob.glob(os.path.join(root, "test") + "/*.*")))
        
        

    def __getitem__(self, index):
        img = np.array(Image.open(self.files[index % len(self.files)]+".jpg"))
        mask = np.array(Image.open(self.files[index % len(self.files)]+".png"))
        
        img = np.concatenate((img,noisy_blur(mask)[:,:,0:1]), axis = 2)
        
        combined = torch.from_numpy(np.concatenate((img,mask),axis=2))
        
        combined = self.transform(combined)

        img_A = combined[:,:,0:4]
        img_B = combined[:,:,4:5]

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
    
    def reg_blur(label = None, threshold = 0.25):
        if label is None: return 1
        if type(label) is not np.ndarray: return 1
        blur = filters.gaussian(label,10,multichannel=True)
        print(np.amax(blur))
        return np.where(np.divide(blur,np.amax(blur))>threshold,1,0)
    
    def noisy_blur(label = None, threshold = 0.25, noise = .25, noise_kernels = 10):
        if label is None: return 1
        if type(label) is not np.ndarray: return 1
        blur = filters.gaussian(label,10,multichannel=True)
        noisetensor = torch.normal(0,1,size=(1,1,noise_kernels,noise_kernels))
        noisetensor = torch.nn.functional.interpolate(noisetensor, size = (len(blur[0]),len(blur[1])), mode = 'bilinear', align_corners = False)
        noisetensor = np.array(noisetensor.permute(2,3,1,0)[:,:,:,0])
        noisetensor = np.divide(noisetensor,np.amax(np.array(noisetensor)))*np.amax(blur)*noise
        blur = blur+noisetensor
        return np.where(np.divide(blur,np.amax(blur))>threshold,1,0)