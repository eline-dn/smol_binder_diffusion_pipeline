#!/bin/bash
#SBATCH --job-name=af2
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=42g
#SBATCH -o log/af2_%A_%a.log
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH -a 1-3


GROUP_SIZE=30 # to edit, here for up to number of arrays * group_size = 90 commands
WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline/b2_1Z9Yout"
cd "$OD/2_af2"

LINES=$(seq -s 'p;' $((($SLURM_ARRAY_TASK_ID-1)*$GROUP_SIZE+1)) $(($SLURM_ARRAY_TASK_ID*$GROUP_SIZE)))
sed -n "${LINES}p" commands_af2 | bash -x
