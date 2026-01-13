#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --partition=l40s
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem 42gb
#SBATCH --time 34:00:00
#SBATCH --output=rfdiffaa2_%A.log

#SBATCH --mail-type=ALL

# Initialise environment and modules
#source /work/lpdi/users/mpacesa/Pipelines/miniforge3/bin/activate /work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/FastBC

/usr/bin/apptainer run --nv rf_se3_diffusion.sif -u run_inference.py inference.ppi_design=True inference.scaffold_guided=False inference.deterministic=False diffuser.T=200 inference.output_prefix=output/1Z9Y/t2_1 inference.input_pdb=input/1Z9Y_clean.pdb contigmap.contigs=[\'70-180,A4-125,A127-260\'] inference.ligand=FUN inference.num_designs=150 inference.design_startnum=0
