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

proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"  # from submodule

CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
}

### Path where the jobs will be run and outputs dumped
WDIR = "/work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout"

if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)

print(f"Working directory: {WDIR}")


DIFFUSION_DIR = f"{WDIR}/0_diffusion"
MPNN_DIR = f"{WDIR}/1_proteinmpnn"

# ----------------1: Running ProteinMPNN on diffused backbones ---------------------------------------------------
#pdbs should be in /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/0_diffusion, ie in DIFFUSION_DIR
#pattern = re.compile(r"t2_\d+_(2|[2-8]\d|89)\.pdb$")
#pattern = re.compile(r"t2_\d+_(1|[3-9]|1[0-9]|9[0-9]|1[0-4][0-9])\.pdb$")
diffused_backbones_good = glob.glob(f"{DIFFUSION_DIR}/*.pdb")
#diffused_backbones_good = [f for f in all_pdbs if pattern.search(f)]

assert len(diffused_backbones_good) > 0, "No good backbones found!"
os.makedirs(MPNN_DIR, exist_ok=True)
os.chdir(MPNN_DIR)

"""the creation of the mask dict from the trb file allow us to use pMPNN on the backbone pdb file from rf diff, only the binder will be redesigned.
Parsing diffusion output TRB files to extract fixed motif residues.
These residues will not be redesigned with proteinMPNN
"""

mask_json_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/make_maskdict_from_trb.py --out masked_pos.jsonl --trb"
for d in diffused_backbones_good:
    mask_json_cmd += " " + d.replace(".pdb", ".trb")

p = subprocess.Popen(mask_json_cmd, shell=True)
(output, err) = p.communicate()
assert os.path.exists("masked_pos.jsonl"), "Failed to create masked positions JSONL file"

MPNN_temperatures = [0.1, 0.2, 0.3]
MPNN_outputs_per_temperature = 5
MPNN_omit_AAs = "CM"

commands_mpnn = []
cmds_filename_mpnn = "commands_mpnn"
with open(cmds_filename_mpnn, "w") as file:
    for T in MPNN_temperatures:
        for f in diffused_backbones_good:
 commands_mpnn.append( ### !!!! here don't forget to change the output folder if needed!
                f"{PYTHON['proteinMPNN']} {proteinMPNN_script} "
                f"--model_type protein_mpnn --ligand_mpnn_use_atom_context 0 --file_ending _T{T} "
                "--fixed_residues_multi masked_pos.jsonl --out_folder ./ " 
                f"--number_of_batches {MPNN_outputs_per_temperature} --temperature {T} "
                f"--omit_AA {MPNN_omit_AAs} --pdb_path {f} "
                f"--checkpoint_protein_mpnn {SCRIPT_DIR}/lib/LigandMPNN/model_params/proteinmpnn_v_48_020.pt\n"
            )
            file.write(commands_mpnn[-1])
print("Number of proteinMPNN commands:", len(commands_mpnn))
print("Example MPNN command:")
print(commands_mpnn[-1])

submit_script = "submit_mpnn.sh"
utils.create_slurm_submit_script(
    filename=submit_script,
    name="1_proteinmpnn",
    mem="4g",
    N_cores=1,
    time="0:45:00",
    partition="h100",
    email=EMAIL,
    array=len(commands_mpnn),
    array_commandfile=cmds_filename_mpnn,
    group=150,
)

p = subprocess.Popen(["sbatch", submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(output, err) = p.communicate()
