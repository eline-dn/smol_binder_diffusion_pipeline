#!/bin/bash
#SBATCH --job-name=3_design_ligMPNN
#SBATCH -t 70:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=4g
#SBATCH -o output_3_design_ligMPNN.log
#SBATCH -e output_3_design_ligMPNN.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH -a 1-4
GROUP_SIZE=125
LINES=$(seq -s 'p;' $((($SLURM_ARRAY_TASK_ID-1)*$GROUP_SIZE+1)) $(($SLURM_ARRAY_TASK_ID*$GROUP_SIZE)))
sed -n "${LINES}p" commands_ligMPNN | bash -x

