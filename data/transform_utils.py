import torch
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import make_grid
import numpy as np
import random
import albumentations as A
import torch
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from torchvision.utils import make_grid
import numpy as np
import random
import albumentations as A
from timm.data.mixup import Mixup
from timm.data.random_erasing import RandomErasing


#---- Augmentations operating on list of images ----#
class ListRandomAugment(object):
    def __init__(self):
        self.transform = T.RandAugment()

    def __call__(self, images):
        for i in range(len(images)):
            images[i] = self.transform(images[i])

        return images


#---- Helper transforms ----#

# Borrowed from https://timm.fast.ai/mixup_cutmix#What-is-Mixup-doing-internally?
def mixup(x, lam):
    """Applies mixup to input batch of images `x`
    
    Args:
    x (torch.Tensor): input batch tensor of shape (bs, 3, H, W)
    lam (float): Amount of MixUp
    """
    x_flipped = x.flip(0).mul_(1-lam)
    x.mul_(lam).add_(x_flipped)
    return x
    

class SIFARMixup:
    def __init__(self, lam=0.8):
        self.transform = mixup
        self.lam = lam

    def __call__(self, images):
        images = self.transform(images, self.lam)

        return images


class SIFARTransform:
    """
    Create super images as proposed in https://arxiv.org/pdf/2106.14104.pdf.
    Specifically, we create a nrow x nrow grid of sequential frames to create a super image.
    """
    def __init__(self, use_sifar, input_size, nrow):
        self.use_sifar = use_sifar
        self.input_size = input_size
        self.nrow = nrow
    
    def __call__(self, x):
        
        # if using SIFAR, x has shape (nrow^2, 3, input_size, input_size) -> (3, nrow*input_size, nrow*input_size)
        if self.use_sifar:  
            x = make_grid(x, nrow=self.nrow, padding=0)
            
        # if not using SIFAR, x has shape (1, 3, input_size, input_size) -> (3, input_size, input_size)
        else:
            x = x.squeeze()
            
        x = F.resize(x, (self.input_size, self.input_size))
        return x


class FlipFrameChanDim:
    def __init__(self):
        pass

    def __call__(self, x):
        # Given a tensor of shape (F, C, H, W), return a tensor of shape(C, F, H, W).
        return x.permute(1, 0, 2, 3)

    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    
class Resize(object):
    def __init__(self, height, width):
        self.h = height
        self.w = width

    def __call__(self, image, target):
        image = T.Resize((self.h, self.w))(image)
        target = T.Resize((self.h, self.w))(target)
        return image, target 

    
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = T.Normalize(mean=self.mean, std=self.std)(image)
        return image, target
#---- -------------------------- ----#


#---- Data augmentation transforms ----#
class RandomChoice(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = random.choice(self.transforms)

    def __call__(self, img):
        return self.t(img)

class RandomHorizontalFlipClassification(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image):
        image = T.RandomHorizontalFlip(p=self.p)(image)
        return image

class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, image, target):

        transform = A.Compose([A.HorizontalFlip(p=self.p)], additional_targets={'image0': 'image'})
        transformed = transform(image=image.numpy(), image0=image.numpy())

        image = transformed['image']
        target = transformed['image0']

        #convert numpy to tensor
        image = torch.from_numpy(image)
        target = torch.from_numpy(target)
        

        return image, target


class ColorJitter(object):
    def __init__(self, brightness, contrast, saturation, hue):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        transform = T.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation, hue=self.hue)
        image = transform(image)
        return image, target

class GaussianBlur(object):
    """
    Gaussian blur augmentation as used in SimCLR https://arxiv.org/abs/2002.05709.
    """
    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image, target):
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        image = T.GaussianBlur(kernel_size=self.kernel_size, sigma=(sigma, sigma))(image)
        return image, target