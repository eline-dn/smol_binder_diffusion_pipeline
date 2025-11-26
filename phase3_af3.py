from Bio.PDB.Polypeptide import is_aa
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


#----------------------------------------------------------------------------------------------------------
"""-----------------------------------------------------SETUP-----------------------------------------------------"""
#----------------------------------------------------------------------------------------------------------

CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    # "af2":"/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
    "ligandMPNN": f"{CONDAPATH}/envs/ligandmpnn_env/bin/python",
    "ColabDesign": "/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "ligandMPNN_relax":f"{CONDAPATH}/envs/ligandmpnn_relax/bin/python"
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
AF3_DIR = f"{WDIR}/4_af3"

DESIGN_DIR_ligMPNNoutput= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/ligMPNN_output"
os.chdir(DESIGN_DIR_ligMPNNoutput)

### prep the json input files
import json

# path to your input JSON
input_path = f"{AF3_DIR}/input/test_1Z9Y_FUN_no_msa_template.json" # template generic json file to use as input for af3


# load JSON
with open(input_path, "r") as f:
    data = json.load(f)

### First collecting MPNN outputs and creating FASTA files for AF2 input
mpnn_fasta = utils.parse_fasta_files(glob.glob(f"{DESIGN_DIR_ligMPNNoutput}/*.fasta"))

mpnn_fasta_clean_half={}
slot=1
count=0
num_seq=len(mpnn_fasta.keys())

for id, seq in mpnn_fasta.items():
  count+=1
  if count > num_seq/4:
    count=0
    slot+=1
  str_slot=str(slot)
  json_dir=f"{AF3_DIR}/input/{str_slot}"
  if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)
  if "seq1" in id:
    continue # keep only seq0 right now and see if we already have binders
  id_clean=id.replace(">","")
  id_clean=id_clean.replace("_seq0\n","")
  id_clean=id_clean.replace("model2_w_ligand_fused_","")
  seq_clean=seq.replace("\n","")
  seq_clean=seq_clean[:-256]
  #mpnn_fasta_clean_half[id_clean]=seq_clean[:-256]
  # modify protein B sequence
  output_path = f"{json_dir}/{id_clean}.json" # binder sequence specific json input for each af3 
  data["name"]=id_clean
  for entry in data["sequences"]:
      if "protein" in entry:
          if entry["protein"]["id"] == ["B"]:
              entry["protein"]["sequence"] = seq_clean
  
  # write modified JSON to a new file
  with open(output_path, "w") as f:
      json.dump(data, f, indent=2)
  


"""
sbatch run_alphafold.sh -i /work/lpdi/users/dobbelst/tools/alphafold3_examples/af_input/fold_input_singleseq.json -o <OUTPUT_DIR> --no-msa
"""
