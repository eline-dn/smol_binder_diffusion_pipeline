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

SCRIPT_DIR=sys.argv[1]
WDIR=sys.argv[2]
# SETUP-----------------------------------------------------
CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    # "af2":"/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
    "ligandMPNN": f"{CONDAPATH}/envs/ligandmpnn_env/bin/python"
    }

### Path where the jobs will be run and outputs dumped
print(f"Working directory: {WDIR}")

# Ligand information
LIGAND = "FUN"

### Setting up design directory and commands
AF2_DIR = f"{WDIR}/2_af2"
#os.makedirs(DESIGN_DIR_ligMPNN+"/logs", exist_ok=True)
MPNN_DIR = f"{WDIR}/1_proteinmpnn"
AF2_DIR = f"{WDIR}/2_af2"
DIFFUSION_DIR = f"{WDIR}/0_diffusion"
good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb") # these models only will be redesigned
if len(good_af2_models)==0:
    raise ValueError("good af2 models not found, check path")

DESIGN_DIR_ligMPNN = f"{WDIR}/3_ligandMPNN"
os.makedirs(DESIGN_DIR_ligMPNN, exist_ok=True)
os.chdir(DESIGN_DIR_ligMPNN)
"""
options /methods:
1) redesign the pMPNN output PDBs, pocket only (a) or whole binder (b)
2) relax and then redesign the pMPNN output PDBs, pocket only (a) or whole binder (b)
3) repredict the complex structure with AF, with a template for the target protein (a) or a template for the whole complex (b), and then redesign the pocket residues with ligandMPNN (or the whole binder)
4) for a given input, redesign pocket with different depths

choosen option:
=> redesign the pMPNN backbones with different distance cutoffs: starting from the pocket only, intermediate redesign and whole binder
"""


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
""" --------------------------------------------3.1 -1.a binding site design with ligandMPNN, redesign the pMPNN outputs, (pocket only on the binder)----------------------------------------------------------------------"""
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Performing 2 design iterations on each input structure
#NSTRUCT = 2

# re-build filtered design name to retrieve them from the pmpnn output dir
good_pmpnn_bb=list()
for design in good_af2_models:
    sub=os.path.basename(design).split("_")
    name="_".join(sub[0:3])+"_"+sub[5]+"_"+sub[3]+".pdb"
    good_pmpnn_bb.append(name)

from Bio.PDB import PDBParser
# extracting the target's fixed residues
parser = PDBParser(QUIET=True)
commands_design = []
cmds_filename_des = "commands_ligMPNN"
with open(cmds_filename_des, "w") as file:
    for pdb in good_pmpnn_bb:
        structure = parser.get_structure("x", f"{MPNN_DIR}/backbones/{pdb}")
        model = structure[0]             # first model
        chain = model["A"]               # chain A
        # count only standard residues
        residues = [res for res in chain.get_residues() if res.id[0] == " "]
        target_reslist=list(map(str,range(len(residues)-256+1,len(residues))))
        #print(pdb +f"native res from the target: {target_reslist[0]}-{target_reslist[-1]}")
        keep_nat=" ".join(target_reslist) # these belong to the target protein and should not be re-designed
        temperatures=" ".join(list(("0.2", "0.3")))
        distance_redesign_cutoffs = " ".join(list(("8.0", "15.0", "500.0")))
        commands_design.append(f"{PYTHON['ligandMPNN']} {SCRIPT_DIR}/scripts/design/simple_redesign.py "
                         f"--pdb {MPNN_DIR}/backbones/{pdb} --redesign_d_cutoff {distance_redesign_cutoffs} --target_positions {keep_nat}"
                         f" --temperature {temperatures} \n" )
        file.write(commands_design[-1])


print("Example design command:")
print(commands_design[-1])
print("Number of commands:")
print(len(commands_design))

"""
### Running design jobs with Slurm.
submit_script = "submit_design.sh"
utils.create_slurm_submit_script(filename=submit_script, name="3_design_ligMPNN", mem="4g", 
                                 N_cores=1, gpu=True, time="70:00:00", array=len(commands_design),
                                 array_commandfile=cmds_filename_des, partition="h100", group=75)

utils.create_slurm_submit_script(filename=submit_script, name="2_af2", mem="6g",
                                      N_cores=2, gpu=True, partition="h100", time="30:00:00", email=EMAIL, array=len(commands_af2),
                                      array_commandfile=cmds_filename_af2, group=25)"""

#p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#(output, err) = p.communicate()

"""