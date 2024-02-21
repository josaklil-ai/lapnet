
import os
import pandas as pd
import numpy as np
import argparse 
import json
from tqdm import tqdm

SUBJECTS = ...
VERBS = ...
TARGETS = ...

UNIQUE_TRIPLETS = ...

with open('subjects.jsonl', 'r') as f:
    SUBJECTS = json.load(f)

with open('verbs.jsonl', 'r') as f:
    VERBS = json.load(f)

with open('targets.jsonl', 'r') as f:
    TARGETS = json.load(f)

_df = pd.read_csv('/pasteur/data/intermountain/video_dataset/FINAL_ANNOTATIONS.txt', sep=' ', header=None)
_df['triplet'] = _df[3].astype(str) + '-' + _df[4].astype(str) + '-' + _df[5].astype(str)
UNIQUE_TRIPLETS = _df['triplet'].unique().tolist()

"""
Feature generating functions. 

Inputs: Frame-level triplet predictions for N videos.
Outputs: N x 456 feature vectors for all videos.

Functions
---------
- create_feature_names: Returns list of feature names.

- get_total_duration: Returns total duration of video.
- get_total_count: Returns total number of triplets in video.
- get_durations_and_counts: Returns duration/counts of each triplet by component.

- get_unique_durations_and_counts: Returns duration/counts of each triplet.

- get_time_to_first_clip (ttfclip): Returns time to first clip.
- get_time_to_first_cut (ttfcut): Returns time to first cut.
- get_port_insertion_length (pilen): Returns total time of port insertion.
- get_idle_time (idle): Returns average idle time of idle.
- get_lhook_before_bleed (lhookbb) : Returns number of l-hook triplets within 10 seconds of bleeding.
- get_grasper_before_bleed (graspbb): Returns number of grasper triplets within 10 seconds of bleeding.
- get_clip_before_bleed (clipbb): Returns number of clip triplets right before bleeding.
- get_clip_after_bleed (clipab): Returns number of clip triplets right after bleeding.
- get_lhook_grasper_gb_pedicle (lhookdiss/graspdiss): Returns number of l-hook/grasper dissections the gallbladder or cystic pedicle right before gallstone or bile spillage.
- get_freq_actions_before_bbg (fabbg): Returns the frequency of actions 1 min before bleeding or bile/gallstone spillage.

- get_clinical_features: Returns clinical features.

"""

def create_feature_names():
    feature_names = []
    feature_names.extend(['vid_id'])
    # feature_names.extend(['tdur', 'tcnt'])

    # for prefix in ['dur', 'cnt']:
    #     for k, v in SUBJECTS.items():
    #         feature_names.extend([f'{prefix}_subject_{int(k)+1}']) 

    #     for k, v in VERBS.items():
    #         feature_names.extend([f'{prefix}_verb_{int(k)+1}'])

    #     for k, v in TARGETS.items():
    #         feature_names.extend([f'{prefix}_target_{int(k)+1}'])

    for prefix in ['dur', 'cnt']:
        for v in UNIQUE_TRIPLETS:
            feature_names.extend([f'{prefix}_unique_{v}'])

    # feature_names.extend(['ttfclip', 'ttfcut', 'pilen', 'idle'])
    # feature_names.extend(['lhookbb', 'graspbb', 'clipbb', 'clipab'])
    # feature_names.extend(['lhookdiss', 'graspdiss', 'fabbg'])
    return feature_names


def get_total_duration(df):
    features = np.zeros(1)
    features[0] = df.loc[len(df) - 1]['Time end '] - df.loc[0]['Time begin ']
    return features


def get_total_count(df):
    features = np.zeros(1)
    features[0] = len(df) - 1
    return features


def get_durations_and_counts(df):
    features = np.zeros((len(SUBJECTS) + len(VERBS) + len(TARGETS)) * 2)

    for k, v in SUBJECTS.items():
        k = int(k)
        for index, row in df.iterrows():
            if int(row['Instrument']) == k:
                features[k] += row['Duration']
                features[k + len(SUBJECTS) + len(VERBS) + len(TARGETS)] += 1
    for k, v in VERBS.items():
        k = int(k)
        for index, row in df.iterrows():
            if int(row['Verb']) == k:
                features[k + len(SUBJECTS)] += row['Duration']
                features[k + 2 * len(SUBJECTS) + len(VERBS) + len(TARGETS)] += 1
    for k, v in TARGETS.items():
        k = int(k)
        for index, row in df.iterrows():
            if int(row['Target']) == k:
                features[k + len(SUBJECTS) + len(VERBS)] += row['Duration']
                features[k + 2 * len(SUBJECTS) + 2 * len(VERBS) + len(TARGETS)] += 1

    return features


def get_unique_durations_and_counts(df):
    features = np.zeros(227 * 2)

    for index, row in df.iterrows():
        triplet = str(row['Instrument']) + '-' + str(row['Verb']) + '-' + str(row['Target'])
        if triplet in UNIQUE_TRIPLETS:
            features[UNIQUE_TRIPLETS.index(triplet)] += row['Duration']
            features[UNIQUE_TRIPLETS.index(triplet) + 227] += 1

    return features


def get_time_to_first_clip(df):
    features = np.zeros(1)
    for index, row in df.iterrows():
        if VERBS[str(row['Verb'])] == 'Clip':
            features[0] = row['Time begin ']
            break
    return features


def get_time_to_first_cut(df):
    features = np.zeros(1)
    for index, row in df.iterrows():
        if VERBS[str(row['Verb'])] == 'Cut':
            features[0] = row['Time begin ']
            break
    return features


def get_port_insertion_length(df):
    features = np.zeros(1)

    first_port_puncture = 0
    last_port_puncture = 0

    first = True
    for i, row in df.iterrows():
        # ensure that we are looking at port insertions in the beginning of the case
        if SUBJECTS[str(row['Instrument'])] == 'Port' and VERBS[str(row['Verb'])] == 'Puncture' and i < len(df) / 4:
            if first:
                first_port_puncture = row['Time begin ']
                first = False
            last_port_puncture = row['Time end ']

    features[0] = last_port_puncture - first_port_puncture
    return features


def get_idle_time(df):
    features = np.zeros(1)
    time_between_actions = []
    for i, row in df.iterrows():
        if i == 0:
            continue
        if row['Time begin '] - df.loc[i - 1]['Time end '] > 0:
            time_between_actions.append(row['Time begin '] - df.loc[i - 1]['Time end '])

    time_between_actions = np.array(time_between_actions)
    features[0] = np.sum(time_between_actions) / len(time_between_actions)
    return features


def get_lhook_before_bleed(df):
    features = np.zeros(1)
    # Find all instances of l-hook 10 seconds before a bleeding
    for i, row in df.iterrows():
        if SUBJECTS[str(row['Instrument'])] == 'L-hook electrocatuery':
            use = row['Time begin ']
            for j, row2 in df.iterrows():
                if j <= i:
                    continue
                if TARGETS[str(row2['Target'])] == 'Blood':
                    if row2['Time begin '] - use <= 10:
                        features[0] += 1
                    else:
                        break
    return features

def get_grasper_before_bleed(df):
    features = np.zeros(1)
    # Find all instances of grasper 10 seconds before a bleeding
    for i, row in df.iterrows():
        if SUBJECTS[str(row['Instrument'])] == 'Grasper':
            use = row['Time begin ']
            for j, row2 in df.iterrows():
                if j <= i:
                    continue
                if TARGETS[str(row2['Target'])] == 'Blood':
                    if row2['Time begin '] - use <= 10:
                        features[0] += 1
                    else:
                        break
    return features


def get_clip_before_bleed(df):
    features = np.zeros(1)
    # Find all instances of clipping right before a bleeding (within 10 seconds)
    for i, row in df.iterrows():
        if VERBS[str(row['Verb'])] == 'Clip':
            use = row['Time begin ']
            for j, row2 in df.iterrows():
                if j <= i:
                    continue
                if TARGETS[str(row2['Target'])] == 'Blood':
                    if row2['Time begin '] - use <= 10:
                        features[0] += 1
                    else:
                        break
    return features


def get_clip_after_bleed(df):
    features = np.zeros(1)
    # Find all instances of clipping right after a bleeding (within 10 seconds)
    for i, row in df.iterrows():
        if VERBS[str(row['Verb'])] == 'Clip':
            use = row['Time begin ']
            for j, row2 in df.iterrows():
                if j > i:
                    break
                if TARGETS[str(row2['Target'])] == 'Blood':
                    if use - row2['Time end '] <= 10:
                        features[0] += 1
    return features


def get_lhook_grasper_gb_pedicle(df):
    features = np.zeros(2)
    for i, row in df.iterrows():
        if (SUBJECTS[str(row['Instrument'])] == 'L-hook electrocatuery' or SUBJECTS[str(row['Instrument'])] == 'Grasper') \
            and (TARGETS[str(row['Target'])] == 'Gallbladder' or TARGETS[str(row['Target'])] == 'Cystic pedicle'):
            use = row['Time begin ']
            for j, row2 in df.iterrows():
                if j <= i:
                    continue
                if TARGETS[str(row2['Target'])] == 'Gallstone' or TARGETS[str(row2['Target'])] == 'Bile':
                    if row2['Time begin '] - use <= 10:
                        if SUBJECTS[str(row['Instrument'])] == 'L-hook electrocatuery':
                            features[0] += 1
                        elif SUBJECTS[str(row['Instrument'])] == 'Grasper':
                            features[1] += 1
                    else:
                        break
    return features


def get_freq_actions_before_bbg(df):
    features = np.zeros(1)
    # Find frequency of actions 1 min before blood/bile/gallstone spillage
    for i, row in df.iterrows():
        if TARGETS[str(row['Target'])] == 'Blood' or TARGETS[str(row['Target'])] == 'Bile' or TARGETS[str(row['Target'])] == 'Gallstone':
            use = row['Time begin ']
            for j, row2 in df.iterrows():
                if j <= i:
                    continue
                if row2['Time begin '] - use <= 60:
                    features[0] += 1
                else:
                    break
    return features


def _preprocess_A(df):
     # Remove note column if it exists
    if 'Note ' in df.columns:
        df = df.drop(columns=['Note ']) 
        
    # Find phase row remove all rows after that if it exists
    if df[df['Instrument'] == 'Event '].shape[0] > 0:
        nan_row = df[df['Instrument'] == 'Event '].index[0]
        df = df.iloc[:nan_row, :]
    return df


def _preprocess_B(df):
    # Convert subject, verb, and target to int using dictionaries
    df['Instrument'] = df['Instrument'].apply(lambda x: list(SUBJECTS.keys())[list(SUBJECTS.values()).index(x)])
    df['Verb'] = df['Verb'].apply(lambda x: list(VERBS.keys())[list(VERBS.values()).index(x)])
    df['Target'] = df['Target'].apply(lambda x: list(TARGETS.keys())[list(TARGETS.values()).index(x)])
    return df


def _preprocess_C(df):
    # Compute durations
    df['Time begin '] = df['Time begin '].astype(float)
    df['Time end '] = df['Time end '].astype(float)
    df['Duration'] = df['Time end '] - df['Time begin ']
    return df


def get_clinical_features(args):
    # Get chole outcomes data 
    root = os.path.dirname(args.pred_dir)
    labels = pd.read_csv(os.path.join(root, 'chole_outcomes.csv'))

    # Drop patient id, procedure, and short date columns
    labels = labels.drop(columns=['Patient ID', 'Procedure', 'Short Date'])
    labels = labels.rename(columns={'Steris ID': 'vid_id'})
    labels['vid_id'] = labels['vid_id'].astype(str)

    # If Assist? contains any of 'PA', 'NP', 'TY', 'Uro', replace with None
    labels['Assist?'] = labels['Assist?'].replace(
        {
            'PA': 'None', 'NP': 'None', 'TY': 'None', 'Uro': 'None', 'R1 uro': 'None',
            'Attg+R3': 'Attg', '2nd attending + R3': 'Attg', '2nd attending': 'Attg', 
            'R5/R2': 'R5', 'R4/R2': 'R4', 'R4+R2': 'R4'
        }
    )

    # Replace <25ml to 25ml and Min to 10ml in EBL column
    labels['EBL'] = labels['EBL'].replace(
        {'<25ml': '25', 'Min': '10', 'min': '10', '<5': '5', '<10': '10', '<20': '20', '<50': '50'}
    )
    labels['EBL'] = labels['EBL'].astype(int)

    # Read in PGS disease severity scores from csv file
    old_pgs = pd.read_csv(os.path.join(root, 'pgs.csv'))
    new_pgs = pd.read_csv(os.path.join(root, 'new_pgs.csv'))
    new_pgs['vid_id'] = new_pgs['Case']
    new_pgs['final'] = new_pgs['Rating (1-5)']
    new_pgs = new_pgs[['vid_id', 'final']]

    pgs = pd.concat([old_pgs, new_pgs], axis=0)
    pgs = pgs.drop_duplicates(subset=['vid_id'])

    pgs['final'] = pgs['final'].replace({'UKNOWN': 0, 'APPENDECTOMY': 0, 'VIDEO NOT FOUND': 0})
    pgs['final'] = pgs['final'].fillna(0)
    pgs['final'] = pgs['final'].astype(int)
    pgs['vid_id'] = pgs['vid_id'].astype(str)

    labels = pd.merge(labels, pgs, on='vid_id')
    return labels


def main(args):
    df = pd.DataFrame(columns=create_feature_names())

    pred_files = os.listdir(args.pred_dir)
    pred_files = [f for f in pred_files if f.endswith('.csv')]
    
    # pred_files = ['/pasteur/data/intermountain/temp_preds/_.csv']

    for pred_file in tqdm(pred_files):
        pred_df = pd.read_csv(os.path.join(args.pred_dir, pred_file))

        pred_df = _preprocess_A(pred_df)
        pred_df = _preprocess_B(pred_df)
        pred_df = _preprocess_C(pred_df)

        features = []
        for func in tqdm([
            # get_total_duration, get_total_count, 
            # get_durations_and_counts, 
            get_unique_durations_and_counts,
            # get_time_to_first_clip, get_time_to_first_cut, get_port_insertion_length, get_idle_time,
            # get_lhook_before_bleed, get_grasper_before_bleed, get_clip_before_bleed, get_clip_after_bleed,
            # get_lhook_grasper_gb_pedicle, get_freq_actions_before_bbg,
        ], leave=False):
            features.extend(func(pred_df))

        case = os.path.basename(pred_file).split('.')[0].split('-')[1]
        df.loc[len(df)] = [case] + features

    # labels = get_clinical_features(args)
    # df = pd.merge(df, labels, on='vid_id')
    df.to_csv(args.outfile, index=False)

    print(df.shape)

    if args.verbose:
        print(df.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, default='/pasteur/data/intermountain/temp_preds', 
                        help='Directory containing frame-level triplet predictions')
    parser.add_argument('--outfile', type=str, default='features.csv',
                        help='Output file name for features')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Prints out additional information')
    args = parser.parse_args()

    main(args)
    print('Done.')