import os, sys, glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import getpass
import subprocess
import time
import importlib
from shutil import copy2
import Bio.PDB

### Path to this cloned GitHub repo:
SCRIPT_DIR = "/work/lpdi/users/eline/smol_binder_diffusion_pipeline"  # edit this to the GitHub repo path. Throws an error by default.
assert os.path.exists(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR + "/scripts/utils")
import utils


# SETUP-----------------------------------------------------
AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"  # from submodule
CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    # "af2":"/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
}

### Path where the jobs will be run and outputs dumped
WDIR = "/work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout"
print(f"Working directory: {WDIR}")

USE_GPU_for_AF2 = True
DIFFUSION_DIR = f"{WDIR}/0_diffusion"
MPNN_DIR = f"{WDIR}/1_proteinmpnn"
AF2_DIR = f"{WDIR}/2_af2"


#--------------------------------------------------2: Running AlphaFold2-------------------------------------------------
os.makedirs(AF2_DIR, exist_ok=True)
os.chdir(AF2_DIR)

## the pdbs need to be trimmed in order to keep only the "binder" part for the pMPNN sequence design (1) and alphafold binder reprediction (2)
fasta_files = glob.glob(f"{MPNN_DIR}/seqs/*.fa") ### output file
output_dir = f"{AF2_DIR}/trimmed_fastas"
os.makedirs(output_dir, exist_ok=True)

for ff in fasta_files:
    with open(ff, "r") as f:
        lines = f.readlines()
    header = ""
    sequence = ""
    output_filename = os.path.join(output_dir, os.path.basename(ff))
    with open(output_filename, "w") as outfile:
        for line in lines:
            if line.startswith(">"):
                if sequence:
                    trimmed_sequence = sequence[:-256]
                    outfile.write(header + trimmed_sequence + "\n")
                header = line.strip() + "\n"
                sequence = ""
            else:
                sequence += line.strip()
        if sequence:
            trimmed_sequence = sequence[:-256]
            outfile.write(header + trimmed_sequence + "\n")
    print(f"Processed and trimmed: {ff} -> {output_filename}")


### First collecting trimmed MPNN outputs and creating FASTA files for AF2 input
mpnn_fasta = utils.parse_fasta_files(glob.glob(f"{AF2_DIR}/trimmed_fastas/*.fa"))
mpnn_fasta = {k: seq.strip() for k, seq in mpnn_fasta.items() if "model_path" not in k}  # excluding the diffused poly-A sequence
# Giving sequences unique names based on input PDB name, temperature, and sequence identifier
mpnn_fasta = {k.split(",")[0]+"_"+k.split(",")[2].replace(" T=", "T")+"_0_"+k.split(",")[1].replace(" id=", ""): seq for k, seq in mpnn_fasta.items()}
print(f"A total of {len(mpnn_fasta)} sequences will be predicted.")
## Splitting the MPNN sequences based on length
## and grouping them in smaller batches for each AF2 job
## Use group size of >40 when running on GPU. Also depends on how many sequences and resources you have.
SEQUENCES_PER_AF2_JOB = 100  # GPU
mpnn_fasta_split = utils.split_fasta_based_on_length(mpnn_fasta, SEQUENCES_PER_AF2_JOB, write_files=True)
## Setting up AlphaFold2 run
AF2_recycles = 3
AF2_models = "4"  # add other models to this string if needed, i.e. "3 4 5"
commands_af2 = []
cmds_filename_af2 = "commands_af2"
with open(cmds_filename_af2, "w") as file:
    for ff in glob.glob("*.fasta"):
        commands_af2.append(f"{PYTHON['af2']} {AF2_script} "
                          f"--af-nrecycles {AF2_recycles} --af-models {AF2_models} "
                          f"--fasta {ff} --scorefile {ff.replace('.fasta', '.csv')}\n")
        file.write(commands_af2[-1])

print("Example AF2 command:")
print(commands_af2[-1])
print("Number of AF2 commands:")
print(len(commands_af2))


submit_script = "submit_af2.sh"
#if USE_GPU_for_AF2 is True:
utils.create_slurm_submit_script(filename=submit_script, name="2_af2", mem="6g",
                                      N_cores=2, gpu=True, partition="h100", time="30:00:00", array=len(commands_af2),
                                      array_commandfile=cmds_filename_af2, group=25) ## don't forget to adjust group!
