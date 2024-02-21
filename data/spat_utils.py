# Define datasets for segmentation branch
# ==============================================================================
import os
import torch
import torchvision.transforms.functional as F
from torchvision.io import read_image
import glob
import pdb
import numpy as np
    

class Cholecseg8k_SpatialDataset(torch.utils.data.Dataset):
    """
    Implementation for CholecSeg8k dataset: https://arxiv.org/pdf/2012.12453.pdf.
    """
    def __init__(self,
                 root_path,
                 transform = None):
        super(Cholecseg8k_SpatialDataset, self).__init__()

        self.root_path = root_path
        
        self.X = sorted(glob.glob(os.path.join(self.root_path, 'imgs/*')))
        self.Y = sorted(glob.glob(os.path.join(self.root_path, 'masks/*')))
        
        self.transform = transform
             
    def __getitem__(self, idx):

        img = read_image(self.X[idx])
        mask = read_image(self.Y[idx])

        # 255 and 0 are corrupted pixels
        classes_dict = {50:0, 11:1, 21:2, 13:3, 12:4, 31:5, 23:6, 24:7, 25:8, 32:9, 22:10, 33:11, 5:12, 255:13, 0:13}

        for k, v in classes_dict.items():
            mask[0,:,:][mask[0,:,:] == k] = v
            mask[1,:,:][mask[1,:,:] == k] = v
            mask[2,:,:][mask[2,:,:] == k] = v

        img = F.convert_image_dtype(img, dtype=torch.float)
    
        if self.transform is not None:
            img, mask = self.transform(img, mask)

        return img, mask[0,:,:]

    def __len__(self):
        return len(self.X)
    
    
class Intermountain_SpatialDataset(torch.utils.data.Dataset):
    """
    In-house Intermountain dataset.
    """
    def __init__(self,
                 root_path: str = '/pasteur/data/intermountain',
                 transform = None):
        super(Intermountain_SpatialDataset, self).__init__()

        self.root_path = root_path
        
        self.X = sorted(glob.glob(os.path.join(self.root_path, 'imgs/*.jpg'), recursive=True))
        self.Y = sorted(glob.glob(os.path.join(self.root_path, 'masks/*.jpg'), recursive=True))
        
        self.transform = transform
             
    def __getitem__(self, idx):
        img = read_image(self.X[idx])
        mask = read_image(self.Y[idx])
        
        if self.transform is not None:
            img, mask = self.transforms(img, mask)
            
        return img, mask

    def __len__(self):
        return len(self.X)