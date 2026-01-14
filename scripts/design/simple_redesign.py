
"""
input: same as light protocol
output: writes the redesigned fasta sequences.
does the same lig MPNN redesign same as the light protocol but b
without the rosetta and scoring part, just raw lig MPNN outputs. (sequences)
"""


# 0.1: Initialisation and setup:
import sys, os, glob, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyrosetta as pyr
import pyrosetta.rosetta
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


import setup_fixed_positions_around_target

# MPNN scripts
SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../../lib/LigandMPNN")
import mpnn_api
from mpnn_api import MPNNRunner

# Utility scripts
sys.path.append(f"{SCRIPT_PATH}/../utils")
import no_ligand_repack
import scoring_utils
import design_utils


# 0.2: Parsing args:

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True, type=str, help="Input PDB")
#parser.add_argument("--params", nargs="+", type=str, help="Params files")
#parser.add_argument("--cstfile", type=str, help="Enzdes constraint file") #can also be None?
#parser.add_argument("--scoring", type=str, required=True, help="Path to a script that implement scoring methods for a particular design job.\n" # use for example scripts/design/scoring/FUN_scoring.py
                    #"Script must implement methods score_design(pose, sfx, catres) and filter_scores(scores), and a dictionary `filters` with filtering criteria.")
#parser.add_argument("--align_atoms", nargs="+", type=str, help="Ligand atom names used for aligning the rotamers. Can also be proved with the scoring script.")
parser.add_argument("--target_positions", nargs="+", type=int, help="Residue positions that belong to the target and should not be redesigned.")
parser.add_argument("--redesign_d_cutoff", nargs="+", required=True, type=float, help ="distance cutoff for determining the pocket residues")
#parser.add_argument("--nstruct", type=int, default=5, help="How many design iterations? (how many output structures per binder)")
parser.add_argument("--temperature",nargs="+", type=float, default=0.2, help="temperature in lig MPNN")
args = parser.parse_args()

INPUT_PDB = args.pdb
#scorefilename = "scorefile.txt"

#N_iter=args.nstruct
temperatures=args.temperature
design_cutoffs=args.redesign_d_cutoff 

"""
Getting PyRosetta started
"""
extra_res_fa = ""

NPROC = os.cpu_count()
if "SLURM_CPUS_ON_NODE" in os.environ:
    NPROC = os.environ["SLURM_CPUS_ON_NODE"]
elif "OMP_NUM_THREADS" in os.environ:
    NPROC = os.environ["OMP_NUM_THREADS"]


DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.gcc" # This binary was compiled on UW systems. It may or may not work correctly on yours
assert os.path.exists(DAB), "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                        "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"


pyr.init(f"{extra_res_fa} -dalphaball {DAB} -beta_nov16 -run:preserve_header -mute all "
         f"-multithreading true -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}")

alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
alpha_3 = ['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
           'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL','GAP']

aa_1_3 = {a:b for a,b in zip(alpha_1,alpha_3)}
aa_3_1 = {b:a for a,b in zip(alpha_1,alpha_3)}

def thread_seq_to_pose(pose, sequence):
    pose2 = pose.clone()
    for i, r in enumerate(sequence):
        if pose.residue(i+1).name1() == r:
            continue
        mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
        mutres.set_target(i+1)
        mutres.set_res_name(aa_1_3[r])
        mutres.apply(pose2)
    return pose2
  
print("Setting up MPNN API")
mpnnrunner = MPNNRunner(model_type="ligand_mpnn", ligand_mpnn_use_side_chain_context=True)  # starting with default checkpoint

# running the relaxation +...
pdb_name = os.path.basename(INPUT_PDB).replace(".pdb", "")
#relaxed_pdb_str=relax_me(INPUT_PDB, f"{pdb_name}relaxed.pdb")

for design_cutoff in design_cutoffs:
    # 1 -  Which residues should or shouldn't be redesigned?
    ###############################################
    ### PARSING PDB AND FINDING POCKET RESIDUES ###
    ###############################################
   
    input_pose = pyrosetta.pose_from_file(INPUT_PDB)
    pose = input_pose.clone()
    ligand_resno = pose.size()
    assert pose.residue(ligand_resno).is_ligand()
    
    matched_residues = design_utils.get_matcher_residues(INPUT_PDB)
    
    _pose2 = pose.clone()
    pdbstr = pyrosetta.distributed.io.to_pdbstring(_pose2)
    print("Identifying positions to redesign, i.e. in the pocket but not from the target")
    pocket_positions = setup_fixed_positions_around_target.get_pocket_positions(pose=_pose2, target_resno=ligand_resno, cutoff_CA=design_cutoff, cutoff_sc=6.0, return_as_list=True) 
    design_res=[]
    target_positions = {int(x) for x in args.target_positions}
    design_list = [
        res.seqpos()
        for res in _pose2.residues
        if (
            res.seqpos() in pocket_positions
            and not res.is_ligand()
            and res.seqpos() not in target_positions
        )
    ]
    
    pr_list="+".join(list(map(str,design_list)))
    print(f"Redesign residues, ie in the pocket but not from the target: {pr_list}")
    #design_list=[res.seqpos() for res in _pose2.residues if res.seqpos() in pocket_positions and not res.is_ligand() and not in target_positions]
    for rn in list(set(design_list)):
                design_res.append(_pose2.pdb_info().chain(rn)+str(_pose2.pdb_info().number(rn))) 

    for temperature in temperatures:
      #########################################################
      ### Running MPNN ####
      #########################################################
      # Setting up MPNN runner 
      inp = mpnnrunner.MPNN_Input()
      inp.pdb =  pdbstr # relaxed_pdb_str
      #inp.fixed_residues = fixed_residues
      inp.redesigned_residues=design_res
      inp.temperature = temperature
      inp.omit_AA = "CM"
      inp.batch_size = 2
      inp.number_of_batches = 1
      print(f"Generating {inp.batch_size*inp.number_of_batches} initial guess sequences with ligandMPNN")
      mpnn_out = mpnnrunner.run(inp)
      with open("sequences.fasta", "a") as f:
        for n, seq in enumerate(mpnn_out["generated_sequences"]):
          #write sequence to fasta file
          print(seq)
          output_name = f"{pdb_name}_lTp{temperature}_dcut{design_cutoff}_seq{n}"
          f.write(f">{output_name}\n{seq}\n")
          # thread sequences to the pdb poses and save as pdb
          input_pose = pyrosetta.pose_from_file(INPUT_PDB)
          pose_threaded=thread_seq_to_pose(input_pose.clone(), seq)
          pose_threaded.dump_pdb(f"{output_name}.pdb")

          
          
print(f"Generated 3 sequences for binder {pdb_name} ")#with temperature {"and".join(temperatures)}, and at redesign cutoffs {"and".join(args.redesign_d_cutoff)} ")
