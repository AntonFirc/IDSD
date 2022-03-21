### Requirements
- TensorFlow 2.3.0
- cuDNN version 7.6
- CUDA version 10.1

### Metacentrum modules

module add anaconda3-2019.10
module add ffmpeg
module add cuda-10.1
module add cudnn-7.6.4-cuda10.0

source activate IDSD

### Running scripts

- `python3 eval_model.py -i .<dataset_path> -m _max_10`
  - `dataset_path` = directory where `mel` and other data directories are located
  
- `python3 train_model.py -i <dataset_path> -n <run_name>`
  - `dataset_path` = directory where `real` and `fake` directories are located, each directory contains spectrograms
  - `run_name` = final model name, used to load/save model weights (saves to, and loads from `./models` directory)
