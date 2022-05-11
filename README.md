# Nine simple tricks deepfake detectors don't want you to know

Beware! All of the code within this repository is for experimental purposes, thus it is not guaranteed to run anywhere you want it to.

### Requirements
- TensorFlow 2.3.0
- cuDNN version 7.6
- CUDA version 10.1

### Running scripts

- `python3 eval_model.py -i .<dataset_path> -m _max_10`
  - `dataset_path` = directory where `mel` and other data directories are located
  
- `python3 train_model.py -i <dataset_path> -n <run_name>`
  - `dataset_path` = directory where `real` and `fake` directories are located, each directory contains spectrograms
  - `run_name` = final model name, used to load/save model weights (saves to, and loads from `./models` directory)

### Trained models

Models are available to download here: https://nextcloud.fit.vutbr.cz/s/8yRcMqxH3nYB6EC 

The model name refers to the used pooling layer and setting: `feature_pooling-layer_pooling-settting.h5`

### Dataset of modified speech

The dataset is a modification of the FoR (for-2-seconds) validation set.
For any information on the original dataset please visit:
 - https://bil.eecs.yorku.ca/datasets/
 - https://ieeexplore.ieee.org/document/8906599

The dataset of modified speech is available here: https://nextcloud.fit.vutbr.cz/s/8yRcMqxH3nYB6EC


### Metacentrum modules

module add anaconda3-2019.10
module add ffmpeg
module add cuda-10.1
module add cudnn-7.6.4-cuda10.0

source activate IDSD