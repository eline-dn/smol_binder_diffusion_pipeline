#!/bin/bash
#SBATCH --job-name=rfdiffusion_array
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem=42gb
#SBATCH --time=34:00:00
#SBATCH --output=rfdiffaa2_%A_%a.log


# setup:
WD="/work/lpdi/users/eline/smol_binder_diffusion_pipeline"
cd $WD

mkdir b2_1Z9Yout/
mkdir b2_1Z9Yout/0_diffusion/
mkdir b2_1Z9Yout/1_protein_mpnn
mkdir b2_1Z9Yout/2_af2
mkdir b2_1Z9Yout/3_ligandMPNN
OD="b2_1Z9Yout/"
ref_pdb="/work/lpdi/users/eline/tools/rf_diffusion_all_atom/input/1Z9Y_clean.pdb"

# 0_ diffusion, 16h per batch
case $SLURM_ARRAY_TASK_ID in
    1)
        echo "running cmd 1 : ppi_design=False, scaffold_guided=False"
        /usr/bin/apptainer run --nv rf_se3_diffusion.sif -u run_inference.py inference.ppi_design=True inference.scaffold_guided=False inference.deterministic=False diffuser.T=200 inference.output_prefix="$OD/0_diffusion/t3_1" inference.input_pdb="$ref_pdb" contigmap.contigs=[\'70-100,A4-125,A127-260\'] inference.ligand=FUN inference.num_designs=150 inference.design_startnum=0
        ;;
    2)
        echo "running cmd 2 : ppi_design=False, scaffold_guided=True"
        /usr/bin/apptainer run --nv rf_se3_diffusion.sif -u run_inference.py inference.ppi_design=False inference.scaffold_guided=True inference.deterministic=False diffuser.T=200 inference.output_prefix="$OD/0_diffusion/t3_2" inference.input_pdb="$ref_pdb" contigmap.contigs=[\'70-100,A4-125,A127-260\'] inference.ligand=FUN inference.num_designs=150 inference.design_startnum=0
        ;;
    3)
        echo "running cmd 3 : ppi_design=True, scaffold_guided=True"
        /usr/bin/apptainer run --nv rf_se3_diffusion.sif -u run_inference.py inference.ppi_design=True inference.scaffold_guided=True inference.deterministic=False diffuser.T=200 inference.output_prefix="$OD/0_diffusion/t3_3" inference.input_pdb="$ref_pdb" contigmap.contigs=[\'70-100,A4-125,A127-260\'] inference.ligand=FUN inference.num_designs=150 inference.design_startnum=0
        ;;
esac

