"""structure:
lighter and (hopefully) faster version of the pocket redesign script
res to keep
res to re design
lig MPNN setup and load pdb
redesign 5 seq / binder
keep the best after  fast scoring with PyRosetta
apply scoring and relax
save pdb with new seq and scores
"""

"""
test: 
/work/lpdi/users/eline/miniconda3/envs/ligandmpnn_env/bin/python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/ligMPNN_light_pocket_design.py --pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_1_20_1_T0.2.pdb --scoring /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/scoring/FUN_scoring.py --redesign_d_cutoff 8.0 --target_positions 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 --nstruct 5
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
parser.add_argument("--target_positions", nargs="+", type=int, help="Residue positions that belong to the target and should not be redesigned.")
parser.add_argument("--redesign_d_cutoff", nargs="+", required=True, type=float, help ="distance cutoff for determining the pocket residues")
parser.add_argument("--nstruct", type=int, default=5, help="How many design iterations? (how many output structures per binder)")
parser.add_argument("--temperature",nargs="+", type=float, default=0.2, help="temperature in lig MPNN")
args = parser.parse_args()

INPUT_PDB = args.pdb
scorefilename = "scorefile.txt"

N_iter=args.nstruct
temperatures=args.temperature
design_cutoffs=args.redesign_d_cutoff 

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


print("Setting up MPNN API")
mpnnrunner = MPNNRunner(model_type="ligand_mpnn", ligand_mpnn_use_side_chain_context=True)  # starting with default checkpoint


for design_cutoff in design_cutoffs:
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
    pdbstr = pyrosetta.distributed.io.to_pdbstring(_pose2) # TO DO remove pyRosetta completely in this step, use biopython insted. the fact that we don't have a param file here shouldn't be a problem since the raw atom coordinate are juste transfered through the pose, 
    # but the relaxation and scoring are just wothless without this params file :(
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
        for N in range(0,N_iter):
            
            # Setting up MPNN runner 
            
            #--redesigned_residues Specifying which residues need to be designed. This example redesigns the first 10 residues while fixing everything else.
            #--batch_size 3 \
            #--number_of_batches 5
            inp = mpnnrunner.MPNN_Input()
            inp.pdb = pdbstr
            #inp.fixed_residues = fixed_residues
            inp.redesigned_residues=design_res
            inp.temperature = temperature
            inp.omit_AA = "CM"
            inp.batch_size = 5
            inp.number_of_batches = 1
            print(f"Generating {inp.batch_size*inp.number_of_batches} initial guess sequences with ligandMPNN")
            mpnn_out = mpnnrunner.run(inp)
            
            
            ##############################################################################
            ### Finding which of the MPNN-packed structures has the best Rosetta score ###
            ##############################################################################
            poses_iter={}
            scores_iter={}
            for n, seq in enumerate(mpnn_out["generated_sequences"]):
                # thraed pose:
                _pose_threaded = design_utils.thread_seq_to_pose(_pose2, seq)
                #_pose_threaded = fix_catalytic_residue_rotamers(_pose_threaded, input_pose, matched_residues) # the catalytic residues, not our case here
                poses_iter[n] = design_utils.repack(_pose_threaded, sfx)
                # score:
                scores_iter[n] = sfx(poses_iter[n]) # apply a score function
                print(f"  Initial sequence {n} total_score: {scores_iter[n]}")
            
            best_score_id = min(scores_iter, key=scores_iter.get)
            # keep the best one:
            _pose = poses_iter[best_score_id].clone()
            #print(f"Relaxing initial guess sequence {best_score_id}")
            # fast relaxation:
            #_pose2 = _pose.clone()
            #fastRelax.apply(_pose2)
            #print(f"Relaxed initial sequence: total_score = {_pose2.scores['total_score']}")
            catalytic_resnos=list()
            ## Applying user-defined custom scoring
            #scores_df = scoring.score_design(_pose2, pyrosetta.get_fa_scorefxn(), catalytic_resnos)
            #filt_scores = scoring.filter_scores(scores_df)
            
            #selecting the best sequence in terms of pyRosetta score:
            
            ####
            ## dumping outputs
            ####
            
            print(f"Doing proper relax and scoring for the best pose: posenumber {best_score_id}")
            good_pose = _pose.clone()
            
            _rlx_st = time.time()
            # skip
            #fastRelax_proper.apply(good_pose)
            print(f"Final relax finished after {(time.time()-_rlx_st):.2f} seconds.")
            
            ## Applying user-defined custom scoring
            scores_df = scoring.score_design(good_pose, pyrosetta.get_fa_scorefxn(), catalytic_resnos)
            sfx(good_pose)
            output_name=f"{pdb_name}_lTp{temperature}_dcut{design_cutoff}_seq{N}"
            scores_df.at[0, "description"] = output_name
        
            print(f"Design iteration {N}, PDB: {output_name}.pdb")
            good_pose.dump_pdb(f"{output_name}.pdb")
            scoring_utils.dump_scorefile(scores_df, scorefilename)
            

print(f"Generated {N_iter} sequences for binder {pdb_name} ")#with temperature {"and".join(temperatures)}, and at redesign cutoffs {"and".join(args.redesign_d_cutoff)} ")
