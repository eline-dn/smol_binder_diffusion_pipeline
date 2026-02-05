#!/bin/bash
#SBATCH --job-name=pmpnn_array
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=60gb
#SBATCH --time=20:00:00
#SBATCH --output=log/pmpnn_%A_%a.log


WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
OD="b2_1Z9Yout/"

cd $WD


DIFFUSION_DIR="$OD/0_diffusion/"

PYTHON="python"  #Python interpreter

cd "$OD/1_protein_mpnn"

# check input files existence
if [ ! -f "masked_pos.jsonl" ]; then
    echo "Erreur : Échec de la création du fichier masked_pos.jsonl !"
    exit 1
fi

if [ ! -f "redesign_pdb.json" ]; then
    echo "Erreur : Échec de la création du fichier redesign_pdb.json !"
    exit 1
fi

# running pMPNN:
conda activate diffusion
#MPNN_temperatures = [0.1, 0.2, 0.3]
case $SLURM_ARRAY_TASK_ID in
    1)
        echo "running cmd 1 "
        python "$WD/lib/LigandMPNN/run.py" --model_type protein_mpnn \
        --ligand_mpnn_use_atom_context 0 \
        --file_ending "_T0.$SLURM_ARRAY_TASK_ID" \
        --fixed_residues_multi masked_pos.jsonl \
        --out_folder ./ \
        --number_of_batches 5 \
        --temperature "0.$SLURM_ARRAY_TASK_ID" \
        --omit_AA "CM" \
        --pdb_path_multi redesign_pdb.json \
        --checkpoint_protein_mpnn "$WD/lib/LigandMPNN/model_params/proteinmpnn_v_48_020.pt"
        ;;
    2)
        echo "running cmd 2 "
        python "$WD/lib/LigandMPNN/run.py" --model_type protein_mpnn \
        --ligand_mpnn_use_atom_context 0 \
        --file_ending "_T0.$SLURM_ARRAY_TASK_ID" \
        --fixed_residues_multi masked_pos.jsonl \
        --out_folder ./ \
        --number_of_batches 5 \
        --temperature "0.$SLURM_ARRAY_TASK_ID" \
        --omit_AA "CM" \
        --pdb_path_multi redesign_pdb.json \
        --checkpoint_protein_mpnn "$WD/lib/LigandMPNN/model_params/proteinmpnn_v_48_020.pt"
        ;;
    3)
        echo "running cmd 3 "
        python "$WD/lib/LigandMPNN/run.py" --model_type protein_mpnn \
        --ligand_mpnn_use_atom_context 0 \
        --file_ending "_T0.$SLURM_ARRAY_TASK_ID" \
        --fixed_residues_multi masked_pos.jsonl \
        --out_folder ./ \
        --number_of_batches 5 \
        --temperature "0.$SLURM_ARRAY_TASK_ID" \
        --omit_AA "CM" \
        --pdb_path_multi redesign_pdb.json \
        --checkpoint_protein_mpnn "$WD/lib/LigandMPNN/model_params/proteinmpnn_v_48_020.pt"
        ;;
esac