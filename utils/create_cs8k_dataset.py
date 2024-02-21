import glob
import os
import cv2
import numpy as np
import argparse


def _write_video_frames(video_root_dir, out_dir):
    """
    Writes frames from each video in video_root_dir that have not already been written.
    
    Args:
        video_root_dir -> root directory of CholecSeg8k dataset. https://www.kaggle.com/datasets/newslab/cholecseg8k
        out_dir -> directory where to write images and masks
    """
    print('Begin to write video frames...')
    
    # Regex for getting image pngs
    train_split_img_regex = os.path.join(video_root_dir, 'video[0, 2-4]*/**/*endo.png')
    val_split_img_regex = os.path.join(video_root_dir, 'video[1, 5]*/**/*endo.png')
    
    # Regex for getting mask pngs
    train_split_mask_regex = os.path.join(video_root_dir, 'video[0, 2-4]*/**/*endo_watershed_mask.png')
    val_split_mask_regex = os.path.join(video_root_dir, 'video[1, 5]*/**/*endo_watershed_mask.png')

    # Collect images and masks
    train_img_paths = sorted(glob.glob(train_split_img_regex))
    train_mask_paths = sorted(glob.glob(train_split_mask_regex))
    print(len(train_img_paths))
    print(len(train_mask_paths))
    val_img_paths = sorted(glob.glob(val_split_img_regex))
    val_mask_paths = sorted(glob.glob(val_split_mask_regex))
    print(len(val_img_paths))
    print(len(val_mask_paths))
    
    train_folder = os.path.join(out_dir, 'train')
    val_folder = os.path.join(out_dir, 'val')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    train_img_folder = os.path.join(train_folder, 'imgs')
    train_mask_folder = os.path.join(train_folder, 'masks')
    os.makedirs(train_img_folder, exist_ok=True)
    os.makedirs(train_mask_folder, exist_ok=True)
    val_img_folder = os.path.join(val_folder, 'imgs')
    val_mask_folder = os.path.join(val_folder, 'masks')
    os.makedirs(val_img_folder, exist_ok=True)
    os.makedirs(val_mask_folder, exist_ok=True)

    print('Writing images for train split...')
    i = 0
    for p in train_img_paths:
        new_loc = os.path.join(train_img_folder, f'img_{i}.png')
        img = cv2.imread(p)
        s = cv2.imwrite(new_loc, img)
        i += 1
        
    print('Writing masks for train split...')
    i = 0
    for p in train_mask_paths:
        new_loc = os.path.join(train_mask_folder, f'mask_{i}.png')
        mask = cv2.imread(p)
        s = cv2.imwrite(new_loc, mask)
        i += 1
        
    print('Writing images for val split...')
    i = 0
    for p in val_img_paths:
        new_loc = os.path.join(val_img_folder, f'img_{i}.png')
        img = cv2.imread(p)
        s = cv2.imwrite(new_loc, img)
        i += 1
        
    print('Writing masks for val split...')
    i = 0
    for p in val_mask_paths:
        new_loc = os.path.join(val_mask_folder, f'mask_{i}.png')
        mask = cv2.imread(p)
        s = cv2.imwrite(new_loc, mask)
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root_dir', type=str, default='/pasteur/data/cholecseg8k',
                        help='Root directory where CholecSeg8k located.')
    parser.add_argument('--out_dir', type=str, default='/pasteur/data/cholecseg8k',
                        help='Directory to store train/val split.')
    args = parser.parse_args()

    _write_video_frames(args.video_root_dir, args.out_dir)
    print(f'Finished creating CholecSeg8k train/val split.')