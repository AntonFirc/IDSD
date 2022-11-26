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

cd IDSD && source activate IDSD

# cd <path to IDSD root>, for example:
cd && python3 IDSD/train.py -i processed_data/for_training_iirt.npy -e processed_data/for_validation_iirt.npy -t processed_data/for_testing_iirt.npy -n for-2sec-iirt

