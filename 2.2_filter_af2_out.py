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

#---------------------------------------------------2.2 Analyzing AF results ---------------------------------------------------

# Combining all CSV scorefiles into one
os.system("head -n 1 $(ls *aa*.csv | shuf -n 1) > scores.csv ; for f in *aa*.csv ; do tail -n +2 ${f} >> scores.csv ; done")
assert os.path.exists("scores.csv"), "Could not combine scorefiles"

## need to trim the reference pdb files to compare binder wth binder, without the target + ligand:
# remove up to aa 410 + ligand at the end (= extract chain A only)
trim_cmd=f"{PYTHON['general']} {SCRIPT_DIR}/scripts/utils/trim_ref_pdb_nterm.py {DIFFUSION_DIR}/ {DIFFUSION_DIR}/filtered_structures/bindersonly "
submit_script = "submit_ref_extraction.sh"
utils.create_slurm_submit_script(filename=submit_script, name="binder_extraction",
                                  mem="16g", N_cores=8, partition="h100", time="0:05:00", email=EMAIL,
                                  command=trim_cmd, outfile_name="output_extraction")

p = subprocess.Popen(["sbatch", submit_script])
(output, err) = p.communicate()


### Calculating the RMSDs of filtered AF2 predictions relative to the diffusion outputs
### Catalytic residue sidechain RMSDs are calculated in the reference PDB has REMARK 666 line present

analysis_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/utils/analyze_af2.py --scorefile scores.csv "\
               f"--ref_path {DIFFUSION_DIR}/bindersonly/ --mpnn --lddt 0.80 --params {' '.join(params)}"
## Running as a Slurm job
submit_script = "submit_af2_analysis.sh"
utils.create_slurm_submit_script(filename=submit_script, name="af2_analysis",
                                     mem="16g", N_cores=8, partition="h100", time="0:60:00", email=EMAIL,
                                     command=analysis_cmd, outfile_name="output_analysis")

p = subprocess.Popen(["sbatch", submit_script])
(output, err) = p.communicate()


###------------- Filtering AF2 scores based on rmsd and plDDT-------------
scores_af2 = pd.read_csv("scores.sc", sep="\s+", header=0)

scores_af2['lDDT'] = pd.to_numeric(scores_af2['lDDT'], errors='coerce')
#scores_af2_filtered = scores_af2.loc[scores_af2['lDDT'] >= 85.0]


### Filtering AF2 scores based on lddt  
scores_af2['rmsd']= pd.to_numeric(scores_af2['rmsd'], errors='coerce')
#scores_af2_filtered=scores_af2[(scores_af2['rmsd'] <= 1.5)]

# cf plots earlier

scores_af2_filtered=scores_af2[(scores_af2['lDDT'] >=85.0) & (scores_af2['rmsd'] <= 1.5)]
utils.dump_scorefile(scores_af2_filtered, "filtered_scores.sc")


### Copying good predictions to a separate directory
os.chdir(AF2_DIR)

if len(scores_af2_filtered) > 0:
    os.makedirs("good", exist_ok=True)
    good_af2_models = [row["Output_PDB"]+".pdb" for idx,row in scores_af2_filtered.iterrows()]
    for pdb in good_af2_models:
        copy2(f"{pdb}", f"good/{pdb}")
    good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb")
else:
    sys.exit("No good models to continue this pipeline with")

os.chdir(f"{AF2_DIR}/good")
