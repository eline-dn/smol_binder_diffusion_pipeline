#!/bin/bash
#SBATCH --job-name=rfdiffusion_array
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=5gb
#SBATCH --time=10:00:00
#SBATCH --output=pmpnn_inputs_%A.log


WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="b2_1Z9Yout/"

cd $WD


DIFFUSION_DIR="$OD/0_diffusion/"

PYTHON="python"  #Python interpreter

cd "$OD/1_proteinmpnn"

# generating masked_pos.jsonl
$PYTHON "$WD/scripts/design/make_maskdict_from_trb.py" \
    --out masked_pos.jsonl \
    --trb "$DIFFUSION_DIR/*.trb"

# check file existence
if [ ! -f "masked_pos.jsonl" ]; then
    echo "Erreur : Échec de la création du fichier masked_pos.jsonl !"
    exit 1
fi

# generating json pdb_path_multi:
find "$DIFFUSION_DIR" -maxdepth 1 -type f -name "*.pdb" -print0 | \
  xargs -0 realpath | \
  jq -R 'rtrimstr("\n")' | \
  jq --slurp 'reduce .[] as $path ({}; . + {($path): ""})' > redesign_pdb.json

# check file existence
if [ ! -f "redesign_pdb.json" ]; then
    echo "Erreur : Échec de la création du fichier masked_pos.jsonl !"
    exit 1
fi

echo "Successfully preparred inputs for pMPNN sequence design"