"""structure:
lighter and (hopefully) faster version of the pocket redesign script
res to keep
res to re design
lig MPNN setup and load pdb
redesign 5 seq / binder
apply scoring
save pdb with new seq and scores
"""
# 0.1: Initialisation and setup:
import sys, os, glob, shutil, subprocess
import numpy as np
import pandas as pd
import seaborn as sns
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
# helper functions:


def get_crude_fastrelax(fastrelax):
    """
    Modifies your fastrelax method to run a very crude relax script
    MonomerRelax2019:
        protocols.relax.RelaxScriptManager: coord_cst_weight 1.0
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.040
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.051
        protocols.relax.RelaxScriptManager: min 0.01
        protocols.relax.RelaxScriptManager: coord_cst_weight 0.5
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.265
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.280
        protocols.relax.RelaxScriptManager: min 0.01
        protocols.relax.RelaxScriptManager: coord_cst_weight 0.0
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.559
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: scale:fa_rep 0.581
        protocols.relax.RelaxScriptManager: min 0.01
        protocols.relax.RelaxScriptManager: coord_cst_weight 0.0
        protocols.relax.RelaxScriptManager: scale:fa_rep 1
        protocols.relax.RelaxScriptManager: repack
        protocols.relax.RelaxScriptManager: min 0.00001
    """
    _fr = fastrelax.clone()
    script = ["coord_cst_weight 1.0",
              "scale:fa_rep 0.1",
              "repack",
              "coord_cst_weight 0.5",
              "scale:fa_rep 0.280",
              "repack",
              "min 0.01",
              "coord_cst_weight 0.0",
              "scale:fa_rep 1",
              "repack",
              "min 0.005",
              "accept_to_best"]
    filelines = pyrosetta.rosetta.std.vector_std_string()
    [filelines.append(l.rstrip()) for l in script]
    _fr.set_script_from_lines(filelines)
    return _fr


def setup_fastrelax(sfx, crude=False):
    fastRelax = pyrosetta.rosetta.protocols.relax.FastRelax(sfx, 1)
    if crude is True:
        fastRelax = get_crude_fastrelax(fastRelax)
    fastRelax.constrain_relax_to_start_coords(True)

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())
    e = pyrosetta.rosetta.core.pack.task.operation.ExtraRotamersGeneric()
    e.ex1(True)
    e.ex1aro(True)
    if crude is False:
        e.ex2(True)
        # e.ex1_sample_level(pyrosetta.rosetta.core.pack.task.ExtraRotSample(1))
    tf.push_back(e)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())
    fastRelax.set_task_factory(tf)
    return fastRelax

def find_nonclashing_rotamers(pose, rotamers, resno, align_atoms):
    print(f"Finding non-clashing rotamers for residue {pose.residue(resno).name()}-{resno}")
    no_clash = []

    clashcheck_time = 0.0
    replace_time = 0.0

    # Pre-calcululating the heavyatoms list
    heavyatoms  = [n for n in range(1, rotamers[0].natoms()+1) if not rotamers[0].atom_is_hydrogen(n)]
    # Re-order heavyatoms based on distance from nbr_atom
    ha_dists = {ha: (rotamers[0].xyz(ha) - rotamers[0].nbr_atom_xyz()).norm() for ha in heavyatoms}
    heavyatoms = sorted(ha_dists, key=ha_dists.get, reverse=True)

    for rotamer in rotamers:
        _st = time.time()
        pose2 = design_utils.replace_ligand_in_pose(pose, rotamer, resno, align_atoms, align_atoms)
        replace_time += time.time()-_st
        _st = time.time()
        if check_bb_clash(pose2, resno, heavyatoms) is False:
            no_clash.append(rotamer)
        clashcheck_time += time.time() - _st
    print(f"Spent {replace_time:.4f}s doing replacements and {clashcheck_time:.4f}s doing clashchecks.")
    print(f"Found {len(no_clash)} non-clashing rotamers.")
    return no_clash


def check_bb_clash(pose, resno, heavyatoms=None):
    """
    Checks if any heavyatom in the defined residue clashes with any backbone
    atom in the pose
    check_bb_clash(pose, resno) -> bool
    Arguments:
        pose (obj, pose)
        resno (int)
    """
    trgt = pose.residue(resno)

    if heavyatoms is None:
        target_heavyatoms = [n for n in range(1, trgt.natoms()+1) if not trgt.atom_is_hydrogen(n)]
        # Re-order heavyatoms based on distance from nbr_atom
        ha_dists = {ha: (trgt.xyz(ha) - trgt.nbr_atom_xyz()).norm() for ha in target_heavyatoms}
        target_heavyatoms = sorted(ha_dists, key=ha_dists.get, reverse=True)
    else:
        target_heavyatoms = heavyatoms

    # Iterating over each heavyatom in the target and checking if it clashes
    # with any backbone atom of any of the neighboring residues
    LIMIT_HA = 2.5
    LIMIT_H = 1.5
    clash = False

    for res in pose.residues:
        if res.seqpos() == resno:
            continue
        if res.is_ligand():
            continue
        if (res.xyz('CA') - trgt.nbr_atom_xyz()).norm() > 14.0:
            continue
        for ha in target_heavyatoms:
            if (res.xyz("CA") - trgt.xyz(ha)).norm() > 5.0:
                continue
            for bb_no in res.all_bb_atoms():
                dist = (trgt.xyz(ha) - res.xyz(bb_no)).norm()
                LIMIT = LIMIT_HA
                if res.atom_is_hydrogen(bb_no):
                    LIMIT = LIMIT_H
                if dist < LIMIT:
                    clash = True
                    break
            if clash is True:
                break
        if clash is True:
            break
    return clash


def check_sc_clash(pose, resno, exclude_residues):
    trgt = pose.residue(resno)
    trgt_heavyatoms = [n for n in range(1,trgt.natoms()+1) if trgt.atom_type(n).element() != "H"]
    LIMIT = 2.0
    clashes = []
    for res in pose.residues:
        if res.seqpos() in exclude_residues:
            continue
        if res.is_ligand():
            continue
        if (res.xyz("CA") - trgt.nbr_atom_xyz()).norm() > 14.0:
            continue
        for ha in trgt_heavyatoms:
            if (trgt.xyz(ha) - res.xyz("CA")).norm() > 10.0:
                continue
            for atomno in range(1, res.natoms()+1):
                if res.atom_type(atomno).element() == "H":
                    continue
                if (trgt.xyz(ha) - res.xyz(atomno)).norm() < LIMIT:
                    clashes.append(res.seqpos())
                    break
            if res.seqpos() in clashes:
                break
    return clashes


def fix_catalytic_residue_rotamers(pose, ref_pose, catalytic_residues):
    _pose = pose.clone()
    mutres = pyrosetta.rosetta.protocols.simple_moves.MutateResidue()
    for resno in catalytic_residues:
        if ref_pose.residue(resno).name() != _pose.residue(resno).name():
            print(f"Fixing catalytic residue {_pose.residue(resno).name()}-{resno} with reference {ref_pose.residue(resno).name()}-{resno}")
            mutres.set_target(resno)
            mutres.set_res_name(ref_pose.residue(resno).name())
            mutres.apply(_pose)
    return _pose



# 0.2: Parsing args:

parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True, type=str, help="Input PDB")
#parser.add_argument("--params", nargs="+", type=str, help="Params files")
#parser.add_argument("--cstfile", type=str, help="Enzdes constraint file") #can also be None?
parser.add_argument("--scoring", type=str, required=True, help="Path to a script that implement scoring methods for a particular design job.\n" # use for example scripts/design/scoring/FUN_scoring.py
                    "Script must implement methods score_design(pose, sfx, catres) and filter_scores(scores), and a dictionary `filters` with filtering criteria.")
#parser.add_argument("--align_atoms", nargs="+", type=str, help="Ligand atom names used for aligning the rotamers. Can also be proved with the scoring script.")
parser.add_argument("--target_positions", nargs="+", type=str, help="Residue positions that belong to the target and should not be redesigned.")
parser.add_argument("--redesign_d_cutoff", required=True, type=float, help ="distance cutoff for determining the pocket residues")
args = parser.parse_args()

INPUT_PDB = args.pdb
scorefilename = "scorefile.txt"



## Loading the user-provided scoring module
sys.path.append(os.path.dirname(args.scoring))
scoring = __import__(os.path.basename(args.scoring.replace(".py", "")))
assert hasattr(scoring, "score_design")
assert hasattr(scoring, "filter_scores")
assert hasattr(scoring, "filters")

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

sfx = pyr.get_fa_scorefxn()

fastRelax = setup_fastrelax(sfx, crude=True)
fastRelax_proper = setup_fastrelax(sfx, crude=False)


# 1 -  Which residues should or shouldn't be redesigned?
###############################################
### PARSING PDB AND FINDING POCKET RESIDUES ###
###############################################
pdb_name = os.path.basename(INPUT_PDB).replace(".pdb", "")
input_pose = pyrosetta.pose_from_file(INPUT_PDB)
pose = input_pose.clone()
ligand_resno = pose.size()
assert pose.residue(ligand_resno).is_ligand()

matched_residues = design_utils.get_matcher_residues(INPUT_PDB)


#########################################################
### Running MPNN ####
#########################################################
_pose2 = pose.clone()
pdbstr = pyrosetta.distributed.io.to_pdbstring(_pose2)
print("Identifying positions to redesign, i.e. in the pocket but not from the target")
pocket_positions = setup_fixed_positions_around_target.get_pocket_positions(pose=_pose2, target_resno=ligand_resno, cutoff_CA=args.redesign_d_cutoff, cutoff_sc=6.0, return_as_list=True) 
design_res=[]
design_list=[res.seqpos() for res in _pose2.residues if res.seqpos() in pocket_positions and not res.is_ligand() and not in target_positions]
for rn in list(set(design_list)):
            design_res.append(_pose2.pdb_info().chain(rn)+str(_pose2.pdb_info().number(rn))) 

print("Setting up MPNN API")
mpnnrunner = MPNNRunner(model_type="ligand_mpnn", ligand_mpnn_use_side_chain_context=True)  # starting with default checkpoint
# Setting up MPNN runner 

#--redesigned_residues Specifying which residues need to be designed. This example redesigns the first 10 residues while fixing everything else.
#--batch_size 3 \
#--number_of_batches 5
inp = mpnnrunner.MPNN_Input()
inp.pdb = pdbstr
#inp.fixed_residues = fixed_residues
inp.redesigned_residues=design_res
inp.temperature = 0.2
inp.omit_AA = "CM"
inp.batch_size = 5
inp.number_of_batches = 2
print(f"Generating {inp.batch_size*inp.number_of_batches} initial guess sequences with ligandMPNN")
mpnn_out = mpnnrunner.run(inp)


##############################################################################
### Finding which of the MPNN-packed structures has the best Rosetta score ###
##############################################################################

for n, seq in enumerate(mpnn_out["generated_sequences"]):
    _pose_threaded = design_utils.thread_seq_to_pose(_pose2, seq)
    #_pose_threaded = fix_catalytic_residue_rotamers(_pose_threaded, input_pose, matched_residues) # the catalytic residues, not our case here
    poses_iter[n] = design_utils.repack(_pose_threaded, sfx)
    scores_iter[n] = sfx(poses_iter[n]) # apply a score function
    print(f"  Initial sequence {n} total_score: {scores_iter[n]}")

best_score_id = min(scores_iter, key=scores_iter.get)
_pose = poses_iter[n].clone()

print(f"Relaxing initial guess sequence {best_score_id}")

_pose2 = _pose.clone()
fastRelax.apply(_pose2)

print(f"Relaxed initial sequence: total_score = {_pose2.scores['total_score']}")

## Applying user-defined custom scoring
scores_df = scoring.score_design(_pose2, pyrosetta.get_fa_scorefxn(), catalytic_resnos)
filt_scores = scoring.filter_scores(scores_df)

results = {N_iter: {"pose": _pose2.clone(), "scores": scores_df.copy()}}

print(f"Iter {N_iter} scores:\n{scores_df.iloc[0]}")
N_iter += 1

    ####
    ## Done iterating, dumping outputs, if any
    ####
    if len(scoring.filter_scores(scores_df)) == 0:
        print(f"Design iteration {N} finished unsuccessfully in {(time.time() - iter_start_time):.2f} seconds.")
        continue

    print(f"Iter {N}, doing final proper relax and scoring")
    good_pose = _pose2.clone()

    _rlx_st = time.time()
    fastRelax_proper.apply(good_pose)
    print(f"Final relax finished after {(time.time()-_rlx_st):.2f} seconds.")

    ## Applying user-defined custom scoring
    scores_df = scoring.score_design(good_pose, pyrosetta.get_fa_scorefxn(), catalytic_resnos)
    sfx(good_pose)
    scores_df.at[0, "description"] = output_name

    print(f"Design iteration {N} finished in {(time.time() - iter_start_time):.2f} seconds.")
    
    if len(scoring.filter_scores(scores_df)) != 0:
        print("Design iteration {N} is successful, dumping PDB: {output_name}.pdb")
        good_pose.dump_pdb(f"{output_name}.pdb")
        scoring_utils.dump_scorefile(scores_df, scorefilename)
