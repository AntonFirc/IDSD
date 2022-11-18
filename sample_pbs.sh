#!/bin/bash
#PBS -N TrainLoopMel
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb:scratch_ssd=32gb:cl_adan=True
#PBS -l walltime=4:00:00
#PBS -m a

module add anaconda3-2019.10
module add ffmpeg
module add cuda/cuda-11.2.0-intel-19.0.4-tn4edsz
module add cudnn/cudnn-8.1.0.77-11.2-linux-x64-intel-19.0.4-wx22b5t

source activate IDSD

# cd <path to IDSD root>, for example:
cd /storage/brno2/home/deemax/IDSD

python3 train.py -i processed_data/for_training_mel.npy -e processed_data/for_eval_mel.npy -t processed_data/for_test_mel.npy -n for-2sec-mel

