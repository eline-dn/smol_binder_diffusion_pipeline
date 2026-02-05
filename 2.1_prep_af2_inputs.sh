#!/bin/bash
#SBATCH --job-name=af2_in
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10gb
#SBATCH --time=01:00:00
#SBATCH --output=af2_inputs_%A.log


WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="b2_1Z9Yout"


# args: WD, OD
python "$WD/2.1_prep_AF2_inputs.py" $WD $OD
cd "$OD/2_af2"
if [ ! -f "commands_af2" ]; then
    echo "Error while creating file commands_af2!"
    exit 1
fi

echo "command file saved in: commands_af2"