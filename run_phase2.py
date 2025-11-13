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

#%load_ext autoreload (google colab stuff)
#%autoreload 2

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


### Python and/or Apptainer executables needed for running the jobs
### Please provide paths to executables that are able to run the different tasks.
### They can all be the same if you have an environment with all of the necessary Python modules in one

CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    # "af2":"/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
}

username = getpass.getuser()  # your username on the running system
EMAIL = "eline.denis@epfl.ch"  # edit based on your organization. For Slurm job notifications.

PROJECT = "CID_1Z9Y"

### Path where the jobs will be run and outputs dumped
WDIR = "/work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout"

if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)

print(f"Working directory: {WDIR}")

USE_GPU_for_AF2 = True


# Ligand information
params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s)
LIGAND = "FUN"

MPNN_DIR = f"{WDIR}/1_proteinmpnn"
AF2_DIR = f"{WDIR}/2_af2"
DIFFUSION_DIR = f"{WDIR}/0_diffusion"

#---------------------------
os.chdir(f"{AF2_DIR}/good")
good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb")
### Aligning the ligand back into the AF2 predictions.
### This is done by aligning the AF2 model to diffusion output and copying over the ligand using PyRosetta.
### --fix_catres option will readjust the rotamer and tautomer of 
### any catalytic residue to be the same as in the reference model.
## need to modify it in order to add the target?

# => maybe directly use the pbd outputs from pMPNN, contains target + binder
align_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/utils/place_ligand_after_af2.py "\
            f"--outdir with_FUN --params {' '.join(params)} --fix_catres "\ # not sure about fix catres
            f"--pdb {' '.join(good_af2_models)} "\
            f"--ref {' '.join(glob.glob(DIFFUSION_DIR+'/*.pdb'))}"
# might alos help:
#parser.add_argument("--align_start", type=int, help="Start position of the alignment region in the reference PDB")
#parser.add_argument("--align_end", type=int,  help="End position of the alignment region in the reference PDB")

p = subprocess.Popen(align_cmd, shell=True)
(output, err) = p.communicate()

## --------------------------------------------3.1 binding site design with ligandMPNN ----------------------------------------------------------------------

### Setting up design directory and commands
os.chdir(WDIR)
DESIGN_DIR_ligMPNN = f"{WDIR}/3.1_design_pocket_ligandMPNN"
os.makedirs(DESIGN_DIR_ligMPNN, exist_ok=True)
os.chdir(DESIGN_DIR_ligMPNN)

AF2_DIR = f"{WDIR}/2_af2"
os.makedirs(DESIGN_DIR_ligMPNN+"/logs", exist_ok=True)

### Performing 5 design iterations on each input structure
NSTRUCT = 10
cstfile = f"{SCRIPT_DIR}/theozyme/HBA/HBA_CYS_UPO.cst" # /!\ need to edit this, provide one that is adapted to the ligand
# can also be none and automatically designed with pyrosetta

commands_design = []
cmds_filename_des = "commands_design"
with open(cmds_filename_des, "w") as file:
    for pdb in glob.glob(f"{AF2_DIR}/good/with_heme/*.pdb"):
        commands_design.append(f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/heme_pocket_ligMPNN.py "
                             f"--pdb {pdb} --nstruct {NSTRUCT} "
                             f"--scoring {SCRIPT_DIR}/scripts/design/scoring/heme_scoring.py " # /!\ probably also needed to change this
                             f"--params {' '.join(params)} --cstfile {cstfile} > logs/{os.path.basename(pdb).replace('.pdb', '.log')}\n")
        file.write(commands_design[-1])

"""
look into these parameters:
parser.add_argument("--keep_native", nargs="+", type=str, help="Residue positions that should not be redesigned. Use 'trb' as argument value to indicate that fixed positions should be taken from the 'con_hal_idx0' list in the corresponding TRB file provided with --trb flag.")
parser.add_argument("--trb", type=str, help="TRB file associated with the input scaffold. Required only when using --keep_native flag.")
parser.add_argument("--design_full", action="store_true", default=False, help="All positions are set designable. Apart from catalytic residues and those provided with --keep_native")
"""

print("Example design command:")
print(commands_design[-1])


### Running design jobs with Slurm.
submit_script = "submit_design.sh"
utils.create_slurm_submit_script(filename=submit_script, name="3.1_design_pocket_ligMPNN", mem="4g", 
                                 N_cores=1, time="3:00:00", email=EMAIL, array=len(commands_design),
                                 array_commandfile=cmds_filename_des)

if not os.path.exists(DESIGN_DIR_ligMPNN+"/.done"):
    p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()

