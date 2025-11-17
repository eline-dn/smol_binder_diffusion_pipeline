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
diffusion_script = "/work/lpdi/users/eline/rf_diffusion_all_atom/run_inference.py"  # edit this
# inpaint_script = "PATH/TO/RFDesign/inpainting/inpaint.py"  # edit this if needed
proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"  # from submodule
AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"  # from submodule
CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    # "af2":"/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
    "ligandMPNN": f"{CONDAPATH}/envs/ligandmpnn_env/bin/python"
    }
PROJECT = "CID_1Z9Y"
### Path where the jobs will be run and outputs dumped
WDIR = "/work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout"
if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)
print(f"Working directory: {WDIR}")
# Ligand information
LIGAND = "FUN"
MPNN_DIR = f"{WDIR}/1_proteinmpnn"
AF2_DIR = f"{WDIR}/2_af2"
DIFFUSION_DIR = f"{WDIR}/0_diffusion"
os.chdir(f"{AF2_DIR}/good")
good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb") # these models only will be redesigned

"""
options /methods:
1) redesign the pMPNN output PDBs, pocket only (a) or whole binder (b)
2) relax and then redesign the pMPNN output PDBs, pocket only (a) or whole binder (b)
3) repredict the complex structure with AF, with a template for the target protein (a) or a template for the whole complex (b), and then redesign the pocket residues with ligandMPNN (or the whole binder)
4) (bonus) for a given input, redesign pocket with different depths

structure:
input structure selection (filter the "good" structures from the pMPNN output file) -> optionnal reprocessing (relaxation or complex reprediction with AF, with ligand?) 
-> choose a set of fixed residues,  and what residues to redesign -> lig_MPNN_design with several iterations to optimize for binders with the best scores defined in the scoring function (pay attention to the Hbonds scoring, adapt to our ligand, also check the stability without ligand function
-> score the outputs

"""
### Setting up design directory and commands
os.chdir(WDIR)
DESIGN_DIR_ligMPNN = f"{WDIR}/3.1_design_pocket_ligandMPNN"
os.makedirs(DESIGN_DIR_ligMPNN, exist_ok=True)
os.chdir(DESIGN_DIR_ligMPNN)

AF2_DIR = f"{WDIR}/2_af2"
os.makedirs(DESIGN_DIR_ligMPNN+"/logs", exist_ok=True)

## --------------------------------------------3.1 -1.a binding site design with ligandMPNN , redesign the pMPNN outputs, pocket only----------------------------------------------------------------------
DESIGN_DIR_ligMPNN_1A= f"{WDIR}/3.1_design_pocket_ligandMPNN/1A"
os.makedirs(DESIGN_DIR_ligMPNN_1A, exist_ok=True)
os.chdir(DESIGN_DIR_ligMPNN_1A)
### Performing 10 design iterations on each input structure
NSTRUCT = 10
cstfile = None #f"{SCRIPT_DIR}/theozyme/HBA/HBA_CYS_UPO.cst" # /!\ need to edit this, provide one that is adapted to the ligand

# re-build filtered design name to retrieve them from the pmpnn output dir
good_pmpnn_bb=list()
for design in good_af2_models:
    sub=os.path.basename(design).split("_")
    name="_".join(sub[0:2])+"_"+sub[5]+"_"+sub[3]+".pdb"
    good_pmpnn_bb.append(name)


commands_design = []
cmds_filename_des = "commands_design"
with open(cmds_filename_des, "w") as file:
    for pdb in good_pmpnn_bb:
        #extract trb:
        sub=os.path.basename(pdb).split("_")
        trb="_".join(sub[0:2])+".trb"
        commands_design.append(f"{PYTHON['ligandMPNN']} {SCRIPT_DIR}/scripts/design/ligMPNN_pocket_design.py "
                             f"--pdb {MPNN_DIR}/backbones/{pdb} --nstruct {NSTRUCT} --keep_native trb --trb {DIFFUSION_DIR}/{trb}" # to indicate some fixed positions
                             f"--scoring {SCRIPT_DIR}/scripts/design/scoring/FUN_scoring.py \n" )# /!\ ligand specific
                             #f"--cstfile {cstfile} > logs/{os.path.basename(pdb).replace('.pdb', '.log')}\n")
        file.write(commands_design[-1])

"""test
/work/lpdi/users/eline/miniconda3/envs/ligandmpnn_env/bin/python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/ligMPNN_pocket_design.py --pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_1_20_1_T0.2.pdb --nstruct 10 --keep_native trb --trb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/0_diffusion/t2_1_20.trb --scoring /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/scoring/FUN_scoring.py
"""
print("Example design command:")
print(commands_design[-1])
### Running design jobs with Slurm.
submit_script = "submit_design.sh"
utils.create_slurm_submit_script(filename=submit_script, name="3.1_design_pocket_ligMPNN", mem="4g", 
                                 N_cores=1, time="30:00:00", array=5,
                                 array_commandfile=cmds_filename_des)

p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(output, err) = p.communicate()

