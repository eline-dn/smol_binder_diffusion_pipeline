from Bio.PDB import PDBIO
import os
import pyrosetta as pyr
import pyrosetta.rosetta
import numpy as np
from pyrosetta.rosetta.core.scoring import fa_rep
import os, sys
import pandas as pd
import pyrosetta.distributed.io
import pyrosetta.rosetta.core.select.residue_selector as residue_selector
import json
import getpass
import argparse
import random
import copy
import time
import scipy.spatial
import io


SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../../utils")

import Bio.PDB
#PDB_parser = Bio.PDB.PDBParser(QUIET=True)
CIF_parser = Bio.PDB.MMCIFParser(QUIET=True)


#---------------------------------------------------------------------------------------------------------------

# Ligand information
params = [f"FUN.params"]  # Rosetta params file(s)
LIGAND = "FUN"


parser = argparse.ArgumentParser()
parser.add_argument("--pdb", nargs="+", type=str, help="Input PDB") # list of pdb files 
#parser.add_argument("--params", nargs="+", type=str, help="Params files")
args = parser.parse_args()


"""
Getting PyRosetta started
"""
extra_res_fa = ""
if True: #args.params is not None:
    extra_res_fa = "-extra_res_fa"
    for p in params:
        extra_res_fa += f" {p}"

NPROC = os.cpu_count()
if "SLURM_CPUS_ON_NODE" in os.environ:
    NPROC = os.environ["SLURM_CPUS_ON_NODE"]
elif "OMP_NUM_THREADS" in os.environ:
    NPROC = os.environ["OMP_NUM_THREADS"]


DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.gcc" # This binary was compiled on UW systems. It may or may not work correctly on yours
assert os.path.exists(DAB), "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                        "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"


pyr.init(f"{extra_res_fa} -dalphaball {DAB} -beta_nov16 -run:preserve_header -mute all ")
        # f"-multithreading false -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}")
df_scores = pd.DataFrame()
selected_df=pd.read_csv("selected2_binder_all_metrics_df.csv")
for i,INPUT_PDB in enumerate(args.pdb): # actually some mmcif files that we convert to pdb for pyrosetta
    structure = CIF_parser.get_structure("x", INPUT_PDB)
    old_id = "FUN"
    new_id = "F"
    chain=structure[0][old_id]
    chain.id = new_id
    for res in structure.get_residues():
        if res.resname.strip().startswith("LIG"):  # or your rule
            res.resname = "FUN"
    io = PDBIO()
    io.set_structure(structure)
    pdbfile=INPUT_PDB.replace(".cif", ".pdb") # convert mmcif to pdb
    io.save(pdbfile)
    
    input_pose = pyrosetta.pose_from_file(pdbfile)
    pose = input_pose.clone()
    ligand_resno = pose.size()
    print("lig pos:",ligand_resno)
    assert pose.residue(ligand_resno).is_ligand()
    
    # Using a custom function to find HBond partners of the groups that might be involved
    # build ligand pose 
    ligand_pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.pose.append_subpose_to_pose(ligand_pose, pose, pose.size(), pose.size(), 1)
   
    ## Calculating shape complementarity between binder and target
    #lig_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(ligand_seqpos)
    target_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
    binder_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("B")
    sc = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
    sc.use_rosetta_radii(True)
    sc.selector1(target_sel)
    sc.selector2(binder_sel)
    df_scores.at[i, "sc"] = sc.score(pose)
    df_scores.at[i,"cif_path"]=INPUT_PDB
    #id_index=selected_df['id']==INPUT_PDB.replace(".cif", "")
    mask = (selected_df['id'] == INPUT_PDB.replace(".cif", ""))
    selected_df.loc[mask, "sc"] = sc.score(pose)
    #selected_df.at[id_index, "sc"] = sc.score(pose)
    #see also: bindcraft's score_interface function in pyrosetta_utils.py
    #interfacescore = iam.get_all_data()
    #interface_sc = interfacescore.sc_value # shape complementarity
    #interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds

df_scores.to_csv("h_scores.csv")
selected_df.to_csv("selected_scores.csv")
