import pandas as pd
import glob
import os
from tqdm import tqdm


def _write_annotation_file(ann_dir, annotation_outfile):
    """
    Writes the annotation file for all triplets for videos in CholecT45.
    """
    triplet_classes = pd.read_csv(os.path.join(os.path.dirname(ann_dir), 'dict/triplet.txt'), sep=',', header=None)

    # Remove everything before colon in first column
    triplet_classes[0] = triplet_classes[0].str.split(':').str[1]

    # Remove -1 from triplet classes
    triplet_classes = triplet_classes[triplet_classes[0] != 'null_instrument']

    # Convert triplet names to integers
    triplet_classes[0] = triplet_classes[0].astype('category')
    triplet_classes[1] = triplet_classes[1].astype('category')
    triplet_classes[2] = triplet_classes[2].astype('category')
    
    triplet_classes[0] = triplet_classes[0].cat.codes
    triplet_classes[1] = triplet_classes[1].cat.codes
    triplet_classes[2] = triplet_classes[2].cat.codes

    # Get total unique categories in each column
    num_subjects = len(triplet_classes[0].unique())
    num_verbs = len(triplet_classes[1].unique())
    num_targets = len(triplet_classes[2].unique())

    # Read all annotation files in ann_dir
    case_txts = []
    case_ids = []
    for txt in glob.glob(os.path.join(ann_dir, '*.txt')):
        case_txts.append(pd.read_csv(txt, sep=',', header=None))
        case_ids.append(os.path.basename(txt))

    # For each case text file, write the triplet annotations to a single file
    print(f'Beginning to write annotation file: {annotation_outfile}.')

    # Save unique triplets
    unique_triplets = set()

    # Get total number of triplets
    num_triplets = 0
    with open(annotation_outfile, 'w') as f:
        for i, case in tqdm(enumerate(case_ids)):

            # Read triplet information for this case
            for j, row in case_txts[i].iterrows():
                if j == 0:
                    continue

                # Get index of all max values in row
                triplet_idxs = [i for i, x in enumerate(row[1:]) if int(x) == 1]
                
                for triplet_idx in triplet_idxs:
                    # Get the row in triplet_classes at triplet_idx
                    triplet = triplet_classes.iloc[triplet_idx]
                    num_triplets += 1

                    # Add unique triplet
                    unique_triplets.add(tuple(triplet))
                    
                    # Get subject, verb, and target
                    subject = triplet[0]
                    verb = triplet[1]
                    target = triplet[2]

                    # Write triplet to file
                    f.write(f'{case[:-4]} {row[0] - 1} {row[0]} {subject} {verb} {target}\n')

    print(f'Wrote {len(unique_triplets)} unique triplets to {annotation_outfile}.')
    print(f'There are {num_triplets} total instances.')
    print(f'There are {num_subjects} subjects, {num_verbs} verbs, and {num_targets} targets.')
    return


if __name__ == '__main__':
    # Set directories
    ann_dir = '/pasteur/data/cholect45/CholecT45/triplet'
    annotation_outfile = '/pasteur/data/cholect45/CholecT45/annotations.txt'

    # Write annotation file
    _write_annotation_file(ann_dir, annotation_outfile)

    print(f'Finished writing annotation file for CholecT45.')