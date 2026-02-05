#!/bin/bash
#SBATCH --job-name=af2_out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=10gb
#SBATCH --time=01:00:00
#SBATCH --output=af2_outputs_%A.log


WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="b2_1Z9Yout/"
DIFFUSION_DIR="$OD/0_diffusion"
AF2_DIR="$OD/2_af2"
mkdir "$DIFFUSION_DIR/bindersonly"

# trim backbone pdbs to align binder to repredicted af2 structure
python "$WD/scripts/utils/trim_ref_pdb_nterm.py" "$DIFFUSION_DIR/" "$DIFFUSION_DIR/bindersonly"
# args: path to the folder with pdb files to trim, and path to the output folder
cd $AF2_DIR
# calculating rmsd af af2 predictions and filtering good models
python "$WD/scripts/utils/analyze_af2.py --scorefile scores.csv \
               --ref_path $DIFFUSION_DIR/bindersonly/ --mpnn --lddt 0.80 --rmsd 1.5"

