"""
from the names of the binders that pass the first round of AF2 filters, get the pMPNN backbone outputs and repredict the complex strucure as a multimer with Colab Design
run relaxation
align the ligand back into the structure 
run lig MPNN without rosetta and scoring stuff on relaxed structure
(saves fasta output)
"""

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
diffusion_script = "/work/lpdi/users/eline/rf_diffusion_all_atom/run_inference.py"  # edit this
proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"  # from submodule
AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"  # from submodule
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
os.chdir(f"{AF2_DIR}/good")
good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb") # these models only will be redesigned
DESIGN_DIR_ligMPNN_alt= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt"
os.makedirs(DESIGN_DIR_ligMPNN_alt, exist_ok=True)
DESIGN_DIR_ligMPNN_alt_af2= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/af2_reprediction"
os.makedirs(DESIGN_DIR_ligMPNN_alt_af2, exist_ok=True)

os.chdir(DESIGN_DIR_ligMPNN_alt)

# get the "good" pMPNN backbones outputs (threaded with seq)
good_pmpnn_bb=list()
for design in good_af2_models:
    sub=os.path.basename(design).split("_")
    name="_".join(sub[0:3])+"_"+sub[5]+"_"+sub[3]+".pdb"
    good_pmpnn_bb.append(name)
 #--------------------------------------------------------------------------------------------------------------------------------------------
"""---------------------------------------------------------------------- repredict structure:-------------------------------------------------------------------------"""
#---------------------------------------------------------------------- 
"""
os.makedirs(f"{DESIGN_DIR_ligMPNN_alt_af2}/monomer", exist_ok=True)
os.chdir(f"{DESIGN_DIR_ligMPNN_alt_af2}/monomer")

commands_reprediction = []
cmds_filename_des = "commands_reprediction"
with open(cmds_filename_des, "w") as file:
    for pdb in good_pmpnn_bb: 
        commands_reprediction.append(f"{PYTHON['ColabDesign']} {SCRIPT_DIR}/scripts/af2/repredict_from_template.py "
                         f"--complex_pdb {MPNN_DIR}/backbones/{pdb}  \n" )
        file.write(commands_reprediction[-1])


print("Example design command:")
print(commands_reprediction[-1])
print("Number of commands:")
print(len(commands_reprediction))
"""
"""test
/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/af2/repredict_from_template.py --complex_pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_1_74_4_T0.2.pdb
"""


### Running design jobs with Slurm.
submit_script = "submit_reprediction.sh"
"""
utils.create_slurm_submit_script(filename=submit_script, name="3.1_reprediction", mem="4g", 
                                 N_cores=1, gpu=True, time="00:250:00",
                                 array_commandfile=cmds_filename_des, array=2, group=223, partition="h100")
"""
""" !!! to add before submission:
source /work/lpdi/users/mpacesa/Pipelines/miniforge3/bin/activate /work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma ; module load gcc/13.2 ; module load cuda/12.4.1 ; module load cudnn/8.9.7.29-12

"""
"""
two options: fixedbb protocol, as a monomer, or binder protocol, as a dimer but using a template for the binder + intial guess + init_atom_pos"""
#--------------------------------------------------------------------------------------------------------------------------------------------
"""----------------------------------------------------------------------(relax?), align structure and put ligand back in:-------------------------------------------------------------------------"""
#----------------------------------------------------------------------
"""
relax it (optionnal)
realign it to the pMPNN backbone
put ligand back in
save for lig MPNN pocket redesign
"""
"""
complex_folder=f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/af2_reprediction/use_init" # where to find the repredicted pdb structures
DESIGN_DIR_ligMPNN_alt_relax= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/with_lig"
os.makedirs(DESIGN_DIR_ligMPNN_alt_relax, exist_ok=True)
os.chdir(DESIGN_DIR_ligMPNN_alt_relax)

commands_relaxation = []
cmds_filename_des = "commands_relaxation"
with open(cmds_filename_des, "w") as file:
    for pdb in glob.glob(f"{complex_folder}/*.pdb"): 
        pdb_bb=os.path.basename(pdb)
        pdb_bb=pdb_bb.replace("_model2.pdb", "")
        commands_relaxation.append(f"{PYTHON['af2']} {SCRIPT_DIR}/scripts/design/relax_align_dimer.py "
                         f"--pdb {pdb} --backbone_pdb {MPNN_DIR}/backbones/{pdb_bb} \n" )
        file.write(commands_relaxation[-1])


print("Example design command:")
print(commands_relaxation[-1])
print("Number of commands:")
print(len(commands_relaxation))


### Running design jobs with Slurm.
submit_script = "submit_relaxation.sh"
utils.create_slurm_submit_script(filename=submit_script, name="3.1_lig", mem="4g", 
                                 N_cores=1, gpu=True, time="00:450:00",
                                 array_commandfile=cmds_filename_des, array=2, group=225, partition="h100")
"""
"""
/work/lpdi/users/eline/miniconda3/envs/mlfold/bin/python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/relax_align_dimer.py --pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/3.1_design_pocket_ligandMPNN/alt/af2_reprediction/use_init/t2_2_33_2_T0.1.pdb_model2.pdb --backbone_pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_2_33_2_T0.1.pdb
"""
#--------------------------------------------------------------------------------------------------------------------------------------------
"""----------------------------------------------------------------------run lig MPNN on structure:-------------------------------------------------------------------------"""
#----------------------------------------------------------------------
NSTRUCT=2
DESIGN_DIR_ligMPNNoutput= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/ligMPNN_output"
DESIGN_DIR_ligMPNN_alt_relax= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/with_lig"
os.makedirs(DESIGN_DIR_ligMPNNoutput, exist_ok=True)
os.chdir(DESIGN_DIR_ligMPNNoutput)


from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)
commands_design = []
cmds_filename_des = "commands_design"
with open(cmds_filename_des, "w") as file:
    for pdb in glob.glob(f"{DESIGN_DIR_ligMPNN_alt_relax}/*.pdb"): ### 
        structure = parser.get_structure("x", pdb) ### 
        model = structure[0]             
        chain = model["A"]               
        # count only standard residues
        residues = [res for res in chain.get_residues() if res.id[0] == " "] #need to have a list of ids with a chain id : A1, A2, ... 
        target_reslist=list(map(str,range(1,len(residues)))) # all the res id that belong to chain A, i.e. to the target
        #print(pdb +f"native res from the target: {target_reslist[0]}-{target_reslist[-1]}")
        keep_nat=" ".join(target_reslist) # these belong to the target protein and should not be re-designed
        temperatures=" ".join(list(("0.2", "0.3")))
        distance_redesign_cutoffs = " ".join(list(("8.0", "15.0", "500.0")))
        commands_design.append(f"{PYTHON['ligandMPNN']} {SCRIPT_DIR}/scripts/design/simple_redesign.py " ### change name of the scipt and the pdbs!!!!
                         f"--pdb {pdb} --nstruct {NSTRUCT} --redesign_d_cutoff {distance_redesign_cutoffs} --target_positions {keep_nat}"
                         f" --temperature {temperatures} \n" )
        file.write(commands_design[-1])


print("Example design command:")
print(commands_design[-1])
print("Number of commands:")
print(len(commands_design))


### Running design jobs with Slurm.
submit_script = "submit_design.sh"
utils.create_slurm_submit_script(filename=submit_script, name="3.1_design_pocket_ligMPNN", mem="4g", 
                                 N_cores=1, gpu=True, time="70:00:00", array=len(commands_design),
                                 array_commandfile=cmds_filename_des, partition="h100", group=75)

"""utils.create_slurm_submit_script(filename=submit_script, name="2_af2", mem="6g",
                                      N_cores=2, gpu=True, partition="h100", time="30:00:00", email=EMAIL, array=len(commands_af2),
                                      array_commandfile=cmds_filename_af2, group=25)"""

p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(output, err) = p.communicate()

