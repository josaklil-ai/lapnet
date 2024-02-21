"""
Make predictions for multiple videos from a checkpoint.
"""
import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import groupby

from data.video_utils import VideoFrameDataset, ImglistToTensor
from data.transform_utils import SIFARTransform
from torchvision import transforms

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


subjects = [
    'Endocatch bag', 
    'Endoloop ligature', 
    'Endoscopic stapler', 
    'Grasper', 
    'Gauze', 
    'Hemoclip', 
    'Hemoclip applier', 
    'Hemostatic agents', 
    'Kittner', 
    'L-hook electrocautery',
    'Maryland dissector', 
    'Needle', 
    'Port', 
    'Scissors', 
    'Suction irrigation', 
    'Unknown instrument'
]

verbs = [
    'Aspirate', 
    'Avulse', 
    'Clip', 
    'Coagulate', 
    'Cut', 
    'Puncture', 
    'Dissect', 
    'Grasp', 
    'Suction/Irrigation', 
    'Pack', 
    'Retract', 
    'Null-verb', 
    'Tear'
]

targets = [
    'Black background', 
    'Abdominal wall', 
    'Adhesion', 
    'Bile', 
    'Blood', 
    'Connective tissue', 
    'Cystic artery',
    'Cystic duct', 
    'Cystic pedicle', 
    'Cystic plate', 
    'Falciform ligament', 
    'Fat', 
    'Gallbladder', 
    'Gallstone', 
    'GI tract', 
    'Hepatoduodenal ligament', 
    'Liver', 
    'Omentum', 
    'Unknown anatomy'
]


def _create_dummy_file(video_path, output_dir, cfg):
    # Create dummy file
    case = os.path.basename(video_path).split('.')[0]
    outfile = os.path.join(output_dir, f'{case}_unif.txt')

    # If dummy file already exists, skip
    if os.path.exists(outfile):
        return outfile
    
    with open(outfile, 'w') as f:
        print(f'Creating dummy sliding window annotation file for case {case}...', end=' ')

        # Get fps for this video
        vidcap = cv2.VideoCapture(video_path)
        fps = vidcap.get(cv2.CAP_PROP_FPS)

        # Get number of frames in video
        num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        S = cfg.datamodule.num_segments * cfg.datamodule.frames_per_segment

        for frame in range(0, num_frames-2*S, S):
            START_FRAME = frame
            END_FRAME = frame + S
            f.write(f'{case} {START_FRAME} {END_FRAME} 0\n')

        print('Done.')
    return outfile


def _create_dummy_dataset(dummy_file, cfg):
    # Create video dataset
    video_dataset = VideoFrameDataset(
        root_path=os.path.join(cfg.temp_dataset.root, 'rgb/'), 
        annotationfile_path=dummy_file,
        num_segments=cfg.datamodule.num_segments, 
        frames_per_segment=cfg.datamodule.frames_per_segment,
        transform=transforms.Compose([
            ImglistToTensor(), 
            transforms.CenterCrop(240),
            SIFARTransform(
                cfg.temp_dataset.use_sifar, 
                cfg.datamodule.temp_img_size,
                int(np.sqrt(cfg.datamodule.num_segments*cfg.datamodule.frames_per_segment)),
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]), 
        test_mode=True,
        imagefile_template=cfg.temp_dataset.img_file_template,
    )

    return DataLoader(
        video_dataset, batch_size=16, shuffle=False, num_workers=8
    )


def _smooth_preds(preds, window_size=5):
    # Smooth predictions with sliding window average
    smoothed_preds = []
    for i in range(len(preds)):
        if i < window_size:
            smoothed_preds.append(preds[i])
        else:
            # Use most frequent prediction in window
            window_preds = torch.tensor(preds[i-window_size:i], dtype=torch.int64)
            smoothed_preds.append(torch.mode(window_preds).values)
    return smoothed_preds


def _preds_to_csv(preds, outfile, fps, threshold=0.5):
    # Loop through predictions and get top-k
    subject_idxs = []
    verb_idxs = []
    target_idxs = []
    for i, pred in enumerate(preds):
        subject_probs = pred[0:len(subjects)]
        verb_probs = pred[len(subjects):len(subjects)+len(verbs)]
        target_probs = pred[len(subjects)+len(verbs):]

        # If the highest probability for any triplet component is less than the threshold, skip
        if (subject_probs.max() < threshold) or (verb_probs.max() < threshold) or (target_probs.max() < threshold):
            subject_idxs.append(-1)
            verb_idxs.append(-1)
            target_idxs.append(-1)
            continue

        # Otherwise, get indices of highest probabilities
        subject_idxs.append(torch.argmax(subject_probs))
        verb_idxs.append(torch.argmax(verb_probs))
        target_idxs.append(torch.argmax(target_probs))

    # Smooth predictions
    window = 5
    subject_idxs = _smooth_preds(subject_idxs, window)
    verb_idxs = _smooth_preds(verb_idxs, window)
    target_idxs = _smooth_preds(target_idxs, window)

    # Loop through predictions again and get triplet names
    triplets = []
    for i, pred in enumerate(preds):
        # If triplet was skipped, skip
        if subject_idxs[i] == -1:
            continue

        # Get subject, verb, and target
        subject = subjects[subject_idxs[i]]
        verb = verbs[verb_idxs[i]]
        target = targets[target_idxs[i]]

        # Create triplet
        triplet = f'{subject},{verb},{target}'
        triplets.append((triplet, i))

    # If same triplet is predicted multiple times in a row, only keep the first instance with start and end frames
    triplet_groups = [list(group) for key, group in groupby(triplets, lambda x: x[0])]
    for i, group in enumerate(triplet_groups):
        if len(group) > 1:
            # Get start and end frames
            start_frame = group[0][1]
            end_frame = group[-1][1]
            triplet_groups[i] = (group[0][0], start_frame, end_frame)
        else:
            triplet_groups[i] = None
    triplet_groups = [x for x in triplet_groups if x is not None]
            
    # Create dataframe
    df = pd.DataFrame(columns=['Instrument', 'Verb', 'Target', 'Time begin ', 'Time end '])
    for triplet in triplet_groups:
        triplet = [triplet[0].split(',')[0], triplet[0].split(',')[1], triplet[0].split(',')[2], triplet[1]/fps, triplet[2]/fps]
        df.loc[len(df)] = triplet

    print(df.head())
    # Save to csv
    df.to_csv(outfile, index=False)


def process_video(gpu_idx, video_path_queue, output_dir, cfg):
    device = torch.device(f'cuda:{gpu_idx}' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = instantiate(cfg.temporal, \
        num_subjects=cfg.temp_dataset.num_subjects, 
        num_verbs=cfg.temp_dataset.num_verbs, 
        num_targets=cfg.temp_dataset.num_targets,
        run_name='',
        log_cfms=False,
    )
    model = model.to(device)
    model.half().eval()

    # Load model
    model.load_state_dict(torch.load(
        map_location=device)['state_dict']
    )

    # Loop through videos with gpu_idx
    video_paths = [x[0] for x in video_path_queue if x[1] == gpu_idx]

    for video_path in video_paths:
        # Create dummy dataset
        dummy_file = _create_dummy_file(video_path, output_dir, cfg)
        dataset = _create_dummy_dataset(dummy_file, cfg)
        outfile = os.path.join(output_dir, os.path.basename(video_path).split('.')[0] + '.csv')

        if len(dataset) == 0:
            case = os.path.basename(video_path).split('.')[0]
            print(f'No frames found for {case} ! Skipping...')
            continue
        if os.path.exists(outfile):
            # or os.path.basename(video_path) == '00491-38258.mp4' \
            #     or os.path.basename(video_path) == '00180-30197.mp4' \
            #         or os.path.basename(video_path) == '00196-27473.mp4':
            continue

        # Get predictions
        preds = []
        for batch in tqdm(dataset):
            inputs, _ = batch
            inputs = inputs.half().to(device)
            with torch.no_grad():
                yhat_s, yhat_v, yhat_t = model(inputs)
                yhat_s = F.softmax(yhat_s, dim=1)
                yhat_v = F.softmax(yhat_v, dim=1)
                yhat_t = F.softmax(yhat_t, dim=1)
                outputs = torch.cat((yhat_s, yhat_v, yhat_t), dim=1)
                preds.append(outputs)
        preds = torch.cat(preds, dim=0)
        # Repeat predictions by 9x
        preds = preds.repeat_interleave(9, dim=0)

        # Get FPS
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Save predictions to csv
        _preds_to_csv(preds, outfile, fps)
        print(f'Done with {os.path.basename(video_path)}!')
        

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    output_dir = '/pasteur/data/intermountain/temp_preds'
    print(f'Saving predictions to {output_dir}...')

    # Get all video paths
    video_paths = glob.glob(os.path.join(os.path.dirname(cfg.temp_dataset.root), '20**/*.mp4'), recursive=True)
    chole_df = pd.read_csv('/pasteur/data/intermountain/chole_outcomes.csv')
    choles = chole_df['Steris ID'].unique().tolist()
    video_paths = [x for x in video_paths if int(os.path.basename(x).split('.')[0].split('-')[1]) in choles]

    # Get device count
    print(f'Predicting on {len(video_paths)}/{len(choles)} video(s) ', end='')
    # n_gpus = torch.cuda.device_count()
    n_gpus=1
    print(f'using {n_gpus} GPU(s)...')

    # Create queue of video paths
    video_path_queue = [(video_path, idx % n_gpus) for idx, video_path in enumerate(video_paths)]

    # Process videos
    mp.spawn(process_video, nprocs=n_gpus, args=(video_path_queue, output_dir, cfg))
    

if __name__ == '__main__':
    main()