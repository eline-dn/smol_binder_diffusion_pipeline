
"""
input: same as light protocol
output: writes the redesigned fasta sequences.
does the same lig MPNN redesign same as the light protocol but b
without the rosetta and scoring part, just raw lig MPNN outputs. (sequences)
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
parser.add_argument("--nstruct", type=int, default=5, help="How many design iterations? (how many output structures per binder)")
parser.add_argument("--temperature",nargs="+", type=float, default=0.2, help="temperature in lig MPNN")
args = parser.parse_args()

INPUT_PDB = args.pdb
#scorefilename = "scorefile.txt"

N_iter=args.nstruct
temperatures=args.temperature
design_cutoffs=args.redesign_d_cutoff 


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
        for N in range(0,N_iter):
            
            # Setting up MPNN runner 
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
            with open("sequences.fasta", "a") as f:
              for n, seq in enumerate(mpnn_out["generated_sequences"]):
                output_name = f"{pdb_name}_lTp{temperature}_dcut{design_cutoff}_seq{n}"
                f.write(f">{output_name}\n{seq}\n")

          
          
print(f"Generated {N_iter} sequences for binder {pdb_name} ")#with temperature {"and".join(temperatures)}, and at redesign cutoffs {"and".join(args.redesign_d_cutoff)} ")
