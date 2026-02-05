#!/bin/bash
#SBATCH --job-name=af2
#SBATCH -t 30:00:00
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem=6g
#SBATCH -o af2_output.log
#SBATCH -e af2_output.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=eline.denis@epfl.ch
#SBATCH -a 1-4

module load gcc/13.2.0
module load cuda/12.4.1
GROUP_SIZE=24 # to edit
WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="b2_1Z9Yout"
cd "$OD/2_af2"

LINES=$(seq -s 'p;' $((($SLURM_ARRAY_TASK_ID-1)*$GROUP_SIZE+1)) $(($SLURM_ARRAY_TASK_ID*$GROUP_SIZE)))
sed -n "${LINES}p" commands_af2 | bash -x

# Combining all CSV scorefiles into one
head -n 1 $(ls *aa*.csv | shuf -n 1) > scores.csv ; for f in *aa*.csv ; do tail -n +2 ${f} >> scores.csv ; done

if [ ! -f "scores.csv" ]; then
    echo "Could not combine scorefiles"
    exit 1
fi
