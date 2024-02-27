
This repo contains code for the [**NEJM AI**](https://ai.nejm.org/) original article "[Artificial Intelligence Identifies Factors Associated with Blood Loss and Surgical Experience in Cholecystectomy](https://ai.nejm.org/doi/full/10.1056/AIoa2300088)". 

### Requirements
```
conda env create -n lapnet -f env.yml
conda activate lapnet
```

### Usage

#### Dataset format
For use on a custom dataset, the surgical video data must be saved as RGB frames on disk in the following format:
```
root
└───video_dataset
    │   annotations.txt
    └───rgb
        │   00370-35490
        |   |   img_000000001.jpg
        |   |   img_000000002.jpg
        |       ...
        |   |   img_000033415.jpg
        │   ...
```
Where `annotations.txt` contains a list of actions with start and stop frames and class labels. This file can be created using the scripts provided in the `utils` folder. One should upload all frames at the available sampling rate into the `rgb` folder (there's also a helpful script for writing video frames to `rgb`).

#### Training
To train the temporal network, run the following command:
```
python run.py logging=True
```
to use WandB logging capabilities. 

To configure data augmentations, enable the available flags by overriding Hydra constants. For example, to enable mixup with alpha 0.8, run the following command:
```
python run.py logging=True temp_dataset.temp_augs_enable_mixup=True temp_dataset.temp_augs_mixup_lam=0.8
```
See the `configs` folder for more details.

#### Evaluation
To evaluate a trained model, run `python eval.py`.

#### Predictions on whole videos
To get the temporal action segmentation predictions on whole surgical videos `python test.py`. This will produce annotations files like the ground truth csvs that would be in `temp_anns`. 

#### Computer vision feature extraction
To extract the surgical activity features used in this study, run `python extract_features.py` (make sure to change the file accordingly given your available clinical metadata).

#### Statistical inference
The code for all statistical analysis found in the paper is in `analysis/code/` in the form of R markdown files. To view the file with proper code and markdown format, we recommend using RStudio.

The extracted features from the CV model is contained in `analysis/data/data.Rda` which is used throughout the statistical analyses. All output .csv that shows the feature effective size and p-values can be found in `analysis/data/output/`

### Citation
```
@article{doi:10.1056/AIoa2300088,
    author = {Josiah G. Aklilu  and Min Woo Sun  and Shelly Goel  and Sebastiano Bartoletti  and Anita Rau  and Griffin Olsen  and Kay S. Hung  and Sophie L. Mintz  and Vicki Luong  and Arnold Milstein  and Mark J. Ott  and Robert Tibshirani  and Jeffrey K. Jopling  and Eric C. Sorenson  and Dan E. Azagury  and Serena Yeung-Levy},
    title = {Artificial Intelligence Identifies Factors Associated with Blood Loss and Surgical Experience in Cholecystectomy},
    journal = {NEJM AI},
    volume = {1},
    number = {2},
    pages = {AIoa2300088},
    year = {2024},
    doi = {10.1056/AIoa2300088},
    URL = {https://ai.nejm.org/doi/abs/10.1056/AIoa2300088},
    eprint = {https://ai.nejm.org/doi/pdf/10.1056/AIoa2300088},
}
```
