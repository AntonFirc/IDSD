#!/bin/bash
#PBS -N TrainLoopMel
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l select=1:ncpus=1:ngpus=1:mem=64gb:scratch_ssd=32gb:cl_adan=True
#PBS -l walltime=4:00:00
#PBS -m a

module add anaconda3-2019.10
module add ffmpeg
module add cuda-10.1
module add cudnn-7.6.4-cuda10.0

source activate IDSD

# cd <path to IDSD root>, for example:
cd /storage/brno2/home/deemax/IDSD

python3 train.py -i processed_data/for_training_mel.npy -e processed_data/for_eval_mel.npy -t processed_data/for_test_mel.npy -n for-2sec-mel

