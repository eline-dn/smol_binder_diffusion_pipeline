#!/bin/bash
#SBATCH --job-name=lig_in
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10gb
#SBATCH --time=01:00:00
#SBATCH --output=lig_inputs_%A.log


WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline/b2_1Z9Yout"

cd $WD
# args: WD, OD
python "$WD/3_lig_MPNN_redesign.py" $WD $OD
cd "$OD/3_ligandMPNN"
if [ ! -f "commands_ligMPNN" ]; then
    echo "Error while creating file commands_ligMPNN!"
    exit 1
fi

echo "command file saved in: commands_ligMPNN"