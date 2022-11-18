# Metacentrum quick-start guide

Primarily work with Adan cluster - tested and optimized for it :)

Clone repo: `git clone https://github.com/AntonFirc/IDSD.git -b spectrogram-generator`

## Interactive job

Ask for computational node: 
`qsub -I -l walltime=4:0:0 -q gpu@meta-pbs.metacentrum.cz -l select=1:ncpus=4:ngpus=1:mem=32gb:scratch_local=64gb:cl_adan=True`
- walltime -> total allowed time to run (hh:mm:ss)
- -q -> queue, do not change
- ncpus, ngpus -> count of CPUs and GPUs
- mem -> RAM
- scratch_local -> local disk space (not needed really)
- cl_adan=True -> specifically reserves only Adan cluster

### Add required modules

- `module add anaconda3-2019.10`
- `module add ffmpeg`
- `module add cuda-10.1`
- `module add cudnn-7.6.4-cuda10.0`

### Conda environment

Create and install requirements only once, then remains usable on all Adan nodes.

- Create: `conda create -n IDSD python==3.8`
- Start: `source activate IDSD`
- Install requirements: `pip install -r requirements.txt`

## Script job

Ideal for training, to let run overnight and longer.

A sample job file in `sample_pbs.sh`