# Image-based Deepfake Speech Detection

Beware! All of the code within this repository is for experimental purposes, thus it is not guaranteed to run anywhere you want it to.

Official implementation of the following paper: Anton Firc, Kamil Malinka, and Petr Hanáček. 2024. Deepfake Speech Detection: A Spectrogram Analysis. In Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing (SAC '24). Association for Computing Machinery, New York, NY, USA, 1312–1320. https://doi.org/10.1145/3605098.3635911

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

The dataset of modified speech is available here: [https://nextcloud.fit.vutbr.cz/s/8yRcMqxH3nYB6EC](https://nextcloud.fit.vutbr.cz/s/oX5MW4XweD4rPeC)


### Metacentrum modules

module add anaconda3-2019.10
module add ffmpeg
module add cuda-10.1
module add cudnn-7.6.4-cuda10.0

source activate IDSD

## Citation
```
@inproceedings{10.1145/3605098.3635911,
author = {Firc, Anton and Malinka, Kamil and Han\'{a}\v{c}ek, Petr},
title = {Deepfake Speech Detection: A Spectrogram Analysis},
year = {2024},
isbn = {9798400702433},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3605098.3635911},
doi = {10.1145/3605098.3635911},
abstract = {The current voice biometric systems have no natural mechanics to defend against deepfake spoofing attacks. Thus, supporting these systems with a deepfake detection solution is necessary. One of the latest approaches to deepfake speech detection is representing speech as a spectrogram and using it as an input for a deep neural network. This work thus analyzes the feasibility of different spectrograms for deepfake speech detection. We compare types of them regarding their performance, hardware requirements, and speed. We show the majority of the spectrograms are feasible for deepfake detection. However, there is no general, correct answer to selecting the best spectrogram. As we demonstrate, different spectrograms are suitable for different needs.},
booktitle = {Proceedings of the 39th ACM/SIGAPP Symposium on Applied Computing},
pages = {1312–1320},
numpages = {9},
keywords = {deepfake, speech, image-based, deepfake detection, spectrogram},
location = {Avila, Spain},
series = {SAC '24}
}
```
