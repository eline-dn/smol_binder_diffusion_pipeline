#!/bin/bash
#SBATCH --job-name=3_design_ligMPNN
#SBATCH -t 70:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32g
#SBATCH -o log/output_3_design_ligMPNN_%A_%a.log
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH -a 1-3
GROUP_SIZE=100
WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline/b2_1Z9Yout"
cd "$OD/3_ligandMPNN"
if [ ! -f "commands_ligMPNN" ]; then
    echo "Could not find file commands_ligMPNN!"
    exit 1
fi

LINES=$(seq -s 'p;' $((($SLURM_ARRAY_TASK_ID-1)*$GROUP_SIZE+1)) $(($SLURM_ARRAY_TASK_ID*$GROUP_SIZE)))
sed -n "${LINES}p" commands_ligMPNN | bash -x

