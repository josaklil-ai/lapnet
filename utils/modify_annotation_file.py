import pandas as pd
import os
from tqdm import tqdm

PATH = '/pasteur/data/intermountain/video_dataset/annotations.txt'

# Read in the annotations file
df = pd.read_csv(PATH, sep=' ', header=None)
print(df.head())
print(f'Number of triplets: {len(df)}')

# If the end of one triplet ends later than the start of the next triplet,
# then change the end of the first triplet to the start of the next triplet,
# or if the end of one triplet ends later than the end of the next triplet,
# then do the same as above but then create a new triplet starting with the end 
# of the second triplet and ending with the end of the first triplet.
# This is done to ensure that there are no overlapping triplets.

num_overlap = 1
while num_overlap > 0:

    # 1. Remove outliers (>2 min)
    for i in range(len(df)):
        if df.iloc[i, 2] - df.iloc[i, 1] > 3600:
            df.iloc[i, 2] = df.iloc[i, 1] + 3600

    # 2. Sort by case and then by start time
    df = df.sort_values(by=[0, 1])

    # 3. Remove overlaps
    num_overlap = 0
    i = 0
    end = len(df) - 1
    for i in tqdm(range(end)):
        j = i+1
        s1, e1 = df.iloc[i][1], df.iloc[i][2]
        s2, e2 = df.iloc[j][1], df.iloc[j][2]

        if df.iloc[i, 0] != df.iloc[j, 0]:
            continue

        assert s1 <= s2, 'Start time of first triplet is greater than start time of second triplet'
        if e1 <= s2:
            continue
        elif e1 > e2:
            if s1 == s2:
                df.iloc[i, 1] = e2 # change start of first triplet to end of second
                num_overlap += 1
                continue
            df.iloc[i, 2] = s2 # change end of first triplet to start of second
            df = pd.concat([df, df.iloc[i:i+1]]).reset_index(drop=True) # create new triplet
            df.iloc[-1, 1] = e2 # change start of new triplet to end of second
            df.iloc[-1, 2] = e1 # change end of new triplet to end of first
            num_overlap += 1
        else:
            df.iloc[i, 2] = s2 # change end of first triplet to start of second
            num_overlap += 1
        i += 1

    # 4. Remove triplets less than 9 frames
    df = df[df[2] - df[1] >= 9]
    print(f'Overlaps: {num_overlap}')
print(f'New number of triplets: {len(df)}')

# Get the total number of frames in the dataset
total_frames = 0
for i in range(len(df)):
    total_frames += df.iloc[i, 2] - df.iloc[i, 1]
print(f'Total number of frames: {total_frames}')

# Test to make sure there are no overlapping triplets
for i in tqdm(range(len(df)-1)):
    s1, e1 = df.iloc[i, 1], df.iloc[i, 2]
    s2, e2 = df.iloc[i+1, 1], df.iloc[i+1, 2]
    if df.iloc[i, 0] != df.iloc[i+1, 0]:
        continue
    assert e1 <= s2, f'Overlap between {i} and {i+1}, case {df.iloc[i, 0]} {s1} {e1} case {df.iloc[i+1, 0]} {s2} {e2}'
    assert e1 - s1 >= 9, f'Length of {i} is less than 9, case {df.iloc[i, 0]} {s1} {e1}'

# Write the triplets to a text file
df.to_csv('/pasteur/data/intermountain/video_dataset/FINAL_ANNOTATIONS.txt', sep=' ', header=False, index=False)