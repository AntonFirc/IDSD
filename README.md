# Nine simple tricks deepfake detectors don't want you to know

Beware! All of the code within this repository is for experimental purposes, thus it is not guaranteed to run anywhere you want it to.

### Requirements
- TensorFlow 2.8.0
- cuDNN version 8.1
- CUDA version 11.2

### Running scripts

- `python3 eval_model.py -i .<dataset_path> -m _max_10`
  - `dataset_path` = directory where `mel` and other data directories are located
  
- `python3 train_generator.py -i <dataset_path> -n <run_name>`
  - `dataset_path` = directory where `real` and `fake` directories are located, each directory contains spectrograms
  - `run_name` = final model name, used to load/save model weights (saves to, and loads from `./models` directory)

### Project data

All data including trained models available here: https://nextcloud.fit.vutbr.cz/s/M6QrWpmatK7fRK3




### Metacentrum modules

module add anaconda3-2019.10
module add ffmpeg
module add cuda/cuda-11.2.0-intel-19.0.4-tn4edsz
module add cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t

source activate IDSD
