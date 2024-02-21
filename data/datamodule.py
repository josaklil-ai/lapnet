from .video_utils import VideoFrameDataset, ImglistToTensor
from .spat_utils import Cholecseg8k_SpatialDataset, Intermountain_SpatialDataset
from .transform_utils import *

import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold



#---- Multi-task Dataset ----#
class MultitaskSurgicalDataset(Dataset):
    def __init__(self, video_dataset, spat_dataset):
        super(MultitaskSurgicalDataset, self).__init__()
        
        self.V = video_dataset
        self.S = spat_dataset
            
    def __getitem__(self, idx):
        return self.V[idx], self.S[idx % len(self.S)]

    def __len__(self):
        return len(self.V)
#---- ------------------ ----#


#----- Datamodule ----#
class SurgicalDatamodule(pl.LightningDataModule):
    """
    Generic datamodule for handling spatial and temporal inputs. 
    """
    def __init__(
        self,
        branch,
        temp_img_size,
        spat_img_size,
        use_sifar,
        temp_root,
        spat_root,
        num_segments,
        frames_per_segment,
        temp_img_file_template,
        temp_batch_size,
        spat_batch_size,
        num_workers,
        temp_augs_enable_center_crop,
        temp_augs_enable_multiscale_jitter,
        temp_augs_enable_mixup,
        temp_augs_enable_cutmix,
        temp_augs_enable_erasing,
        temp_augs_enable_augmix,
        temp_augs_enable_normalize,
        temp_augs_use_sifar
    ):
        super().__init__()

        self.branch = branch     # what branch of the multi-task network to activate
        self.temp_img_size = temp_img_size
        self.spat_img_size = spat_img_size
        self.use_sifar = use_sifar
        self.temp_root = temp_root
        self.spat_root = spat_root
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.temp_img_file_template = temp_img_file_template
        self.temp_batch_size = temp_batch_size
        self.spat_batch_size = spat_batch_size
        self.num_workers = num_workers
        self.temp_augs_enable_center_crop = temp_augs_enable_center_crop
        self.temp_augs_enable_multiscale_jitter = temp_augs_enable_multiscale_jitter
        self.temp_augs_enable_mixup = temp_augs_enable_mixup
        self.temp_augs_enable_cutmix = temp_augs_enable_cutmix
        self.temp_augs_enable_erasing = temp_augs_enable_erasing
        self.temp_augs_enable_augmix = temp_augs_enable_augmix
        self.temp_augs_enable_normalize = temp_augs_enable_normalize
        self.temp_augs_use_sifar = temp_augs_use_sifar

        temp_augs_mixup_lam = 0.8
        temp_augs_cutmix_prob = 1.0
        temp_augs_erasing_prob = 0.25
        temp_augs_augmix_severity = 3
        nrow = 3
        

        # Transformations for spatial inputs
        if self.branch == 'spatial' or self.branch == 'both':
            #print("applying spatial transformations")
            self._train_cholec_augmentations = Compose([
                Resize(self.spat_img_size, self.spat_img_size),
                RandomHorizontalFlip(p = 0.5),
                #ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                #GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self._test_cholec_augmentations = Compose([
                Resize(self.spat_img_size, self.spat_img_size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # Transformations for temporal inputs.
        if self.branch == 'temporal' or self.branch == 'both':
            
            print("Augmentations for temporal inputs:")
            print("temp_augs_enable_center_crop = {}".format(temp_augs_enable_center_crop))
            print("temp_augs_enable_multiscale_jitter = {}".format(temp_augs_enable_multiscale_jitter))
            print("temp_augs_enable_mixup = {}".format(temp_augs_enable_mixup))
            print("temp_augs_enable_cutmix = {}".format(temp_augs_enable_cutmix))
            print("temp_augs_enable_erasing = {}".format(temp_augs_enable_erasing))
            print("temp_augs_enable_augmix = {}".format(temp_augs_enable_augmix))
            print("temp_augs_enable_normalize = {}".format(temp_augs_enable_normalize))
            print("temp_augs_use_sifar = {}".format(temp_augs_use_sifar))

            ilt = ImglistToTensor()
            cwc = transforms.CenterCrop(240) if self.temp_augs_enable_center_crop else None
            msj = transforms.Compose([
                transforms.Resize(240), # 256
                transforms.RandomResizedCrop((224, 224), scale=(0.65, 1.0)), # [(256), 224, 192, 168]
            ]) if self.temp_augs_enable_multiscale_jitter else None
            mixup = SIFARMixup(temp_augs_mixup_lam) if self.temp_augs_enable_mixup else None
            #cutmix = SIFARCutmix(temp_augs_cutmix_prob) if temp_augs_enable_cutmix else None
            erase = transforms.RandomErasing(temp_augs_erasing_prob) if self.temp_augs_enable_erasing else None
            augmix = transforms.AugMix(temp_augs_augmix_severity) if temp_augs_enable_augmix else None
            sifar = SIFARTransform(self.temp_augs_use_sifar, temp_img_size, nrow)
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if self.temp_augs_enable_normalize else None

            # Compose transforms that are not None
            train_transforms = [t for t in [ilt, cwc, msj, mixup, erase, augmix, sifar, norm] if t is not None]
            test_transforms = [t for t in [ilt, cwc, sifar, norm] if t is not None]

            self._train_SIFAR_augmentations = transforms.Compose(train_transforms)
            self._test_SIFAR_augmentations = transforms.Compose(test_transforms)


    def prepare_data(self):
        """
        Nothing to download or tokenize for now, unless clinical notes need to be parsed.
        """
        pass

    def setup(self, stage=None):
        """
        Specifiy location of images and masks for the spatial branch, and videos and temporal annotations for 
        the temporal branch. Also assign train/val splits for use in dataloaders.
        """
        
        if self.branch == 'temporal' or self.branch == 'both':
            video_root = os.path.join(self.temp_root, 'rgb/')
            # annotation_file = os.path.join(self.temp_root, 'annotations.txt')
            annotation_file = os.path.join(self.temp_root, 'FINAL_ANNOTATIONS.txt')

            # Split videos into train/val/test
            with open(annotation_file, 'r') as f:
                lines = f.readlines()
                lines = [line.strip().split(' ') for line in lines]
                lines = np.array([
                    [line[0], int(line[1]), int(line[2]), int(line[3]), int(line[4]), int(line[5])]
                    for line in lines
                ])

                # Split the action triplets intro train/val but preserve split on patient ID
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
                train_idx, val_idx = next(cv.split(lines[:, 1:3], lines[:, 4], lines[:, 0])) # only stratify by verbs

                train_lines = [lines[i] for i in train_idx]
                val_lines = [lines[i] for i in val_idx]

                try:
                    os.remove(os.path.join(self.temp_root, 'train_annotations.txt'))
                    os.remove(os.path.join(self.temp_root, 'val_annotations.txt'))
                except:
                    print('No train/val split exists.')

                # Write the new annotation files
                with open(os.path.join(self.temp_root, 'train_annotations.txt'), 'w') as f:
                    for line in train_lines:
                        f.write(' '.join([str(x) for x in line]) + '\n')

                with open(os.path.join(self.temp_root, 'val_annotations.txt'), 'w') as f:
                    for line in val_lines:
                        f.write(' '.join([str(x) for x in line]) + '\n')

            num_segments = self.num_segments
            frames_per_segment = self.frames_per_segment

            if self.use_sifar:
                num_frames = num_segments * frames_per_segment
                assert (np.sqrt(num_frames).is_integer()) , "SIFAR expects n sequential frames that can be made into a square grid."

            # Assign train/test datasets for videos
            self.video_train_dataset = VideoFrameDataset(
                root_path=video_root, annotationfile_path=os.path.join(self.temp_root, 'train_annotations.txt'),
                num_segments=num_segments,
                frames_per_segment=frames_per_segment,
                transform=self._train_SIFAR_augmentations, 
                test_mode=False,
                imagefile_template=self.temp_img_file_template,
            )

            self.video_val_dataset = VideoFrameDataset(
                root_path=video_root, annotationfile_path=os.path.join(self.temp_root, 'val_annotations.txt'),
                num_segments=num_segments,
                frames_per_segment=frames_per_segment,
                transform=self._test_SIFAR_augmentations,
                test_mode=True,
                imagefile_template=self.temp_img_file_template,
            )
            
        if self.branch == 'spatial' or self.branch == 'both':
            self.frame_train_dataset = Cholecseg8k_SpatialDataset(
                os.path.join(self.spat_root, 'train'), self._train_cholec_augmentations
            )
            self.frame_val_dataset = Cholecseg8k_SpatialDataset(
                os.path.join(self.spat_root, 'val'), self._test_cholec_augmentations
            )
        
        # Create train/val datasets for dataloaders
        if self.branch == 'both':
            self.train_dataset = MultitaskSurgicalDataset(self.video_train_dataset, self.frame_train_dataset)
            self.val_dataset = MultitaskSurgicalDataset(self.video_val_dataset, self.frame_val_dataset)
            self.batch_size = self.temp_batch_size
            
        elif self.branch == 'spatial':
            self.train_dataset = self.frame_train_dataset
            self.val_dataset = self.frame_val_dataset
            self.batch_size = self.spat_batch_size
            
        elif self.branch == 'temporal':
            self.train_dataset = self.video_train_dataset
            self.val_dataset = self.video_val_dataset
            self.batch_size = self.temp_batch_size
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True
        )

    def test_dataloader(self):
        pass
#---- ------------------ ----#


if __name__ == '__main__':
    pass