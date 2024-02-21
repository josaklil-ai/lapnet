import pandas as pd
import glob
import os
import datetime
import cv2


def _read_annotation_csvs(ann_dir):
    """
    Reads annotation csv file for each case in ann_dir and returns list of dataframes.
    """
    csv_paths = glob.glob(os.path.join(ann_dir, '*.csv'))
    case_ids = [os.path.basename(csv_path).split('.')[0] + '.mp4' for csv_path in csv_paths]
    case_csvs = [pd.read_csv(csv_path) for csv_path in csv_paths]
    print(f'Found {len(case_ids)} cases in annotation root: {ann_dir}.')
    return case_ids, case_csvs


def _write_annotation_file(video_root_dir, annotation_outfile, case_ids, case_csvs):
    """
    Writes the annotation file for all triplets from labeled videos in video_root_dir.
    """
    all_video_paths = glob.glob(os.path.join(video_root_dir, '**/*.mp4'))
    written_video_paths = glob.glob(os.path.join(video_root_dir, 'video_dataset/rgb/*'))
    written_video_paths = [os.path.basename(written_video_path) for written_video_path in written_video_paths]

    print(f'Total number videos: {len(all_video_paths)}.')
    print(f'Videos written to disk: {len(written_video_paths)}.')
    
    unique_triplets = []

    # For multi-label procuration
    subjects = ['Endocatch bag', 'Endoloop ligature', 'Endoscopic stapler', 'Grasper', 'Gauze', 
    'Hemoclip', 'Hemoclip applier', 'Hemostatic agents', 'Kittner', 'L-hook electrocautery',
    'Maryland dissector', 'Needle', 'Port', 'Scissors', 'Suction irrigation', 'Unknown instrument']
    verbs = ['Aspirate', 'Avulse', 'Clip', 'Coagulate', 'Cut', 'Puncture', 'Dissect', 'Grasp', 
    'Suction/Irrigation', 'Pack', 'Retract', 'Null-verb', 'Tear']
    targets = ['Black background', 'Abdominal wall', 'Adhesion', 'Bile', 'Blood', 'Connective tissue', 'Cystic artery',
    'Cystic duct', 'Cystic pedicle', 'Cystic plate', 'Falciform ligament', 'Fat', 'Gallbladder', 'Gallstone', 'GI tract', 
    'Hepatoduodenal ligament', 'Liver', 'Omentum', 'Unknown anatomy']
    
    print(f'Beginning to write annotation file: {annotation_outfile}.')
    os.remove(annotation_outfile)
    with open(annotation_outfile, 'w') as f:
        for i, case in enumerate(case_ids):
            case_path = None

            # Search for full path name of annotated case 
            for path in all_video_paths:
                if case == os.path.basename(path):
                    case_path = path
                
            # If the case is not in your video root directory, skip it
            #valid_cases = set(["00423-36845","00411-36435","00084-30294","00011-23317","00160-31801","00628-41813","00659-42988","00392-36036","00122-30748","00523-38955","00225-31889","00069-29951"])
            
            valid_cases = set(["00158-30604", "00346-34641", "00569-39646"])
            if case_path is None or case[:-4] not in written_video_paths or case[:-4] in valid_cases:
                print(f'Case {case} not found in video root directory. Skipping...')
                continue
            else:
                print(f'Writing annotations for case {case}...')
                num_files = len([name for name in os.listdir(f'/pasteur/data/intermountain/video_dataset/rgb/{case[:-4]}')])

                if num_files == 0:
                    print(f'Case {case} has no frames. Skipping...')
                    continue


            # Get fps for this video
            vidcap = cv2.VideoCapture(case_path)
            fps = vidcap.get(cv2.CAP_PROP_FPS)
                

            # Read triplet information for this case
            for _, row in case_csvs[i].iterrows():
                if type(row['Target']) == str:
                    
                    # Collect one triplet and its seconds of start/finish
                    triplet = row['Instrument'] + ', ' +  row['Verb'] + ', ' + row['Target']

                    tb = pd.to_datetime(row['Time begin '])
                    te = pd.to_datetime(row['Time end '])
                    START_FRAME = int((tb.second + tb.minute*60 + tb.hour*3600) * fps)
                    END_FRAME = int((te.second + te.minute*60 + te.hour*3600) * fps)

                    # Check if this type of triplet has been seen before
                    if triplet not in unique_triplets:
                        unique_triplets.append(triplet)

                    # Get the class index of the triplet
                    action_class = unique_triplets.index(triplet)
                    subject_class = subjects.index(row['Instrument'])
                    verb_class = verbs.index(row['Verb'])
                    target_class = targets.index(row['Target'])
                
                    # Write annotation of triplet to text file
                    f.write(f'{case[:-4]} {START_FRAME} {END_FRAME} {subject_class} {verb_class} {target_class}\n')

    print(f'Wrote {len(unique_triplets)} unique triplets to {annotation_outfile}.')
    return


if __name__ == '__main__':
    # Set directories
    video_root_dir = '/pasteur/data/intermountain'
    ann_dir = '/pasteur/data/intermountain/temp_anns'
    annotation_outfile = '/pasteur/data/intermountain/video_dataset/annotations.txt'

    # Read annotation csv files
    case_ids, case_csvs = _read_annotation_csvs(ann_dir)

    # Write annotation file
    _write_annotation_file(video_root_dir, annotation_outfile, case_ids, case_csvs)

    print(f'Finished writing annotation file.')