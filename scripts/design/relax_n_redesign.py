
"""
input: same as light protocol
output: writes the redesigned fasta sequences.
does the same lig MPNN redesign same as the light protocol but b
without the rosetta and scoring part, just raw lig MPNN outputs. (sequences)
the input pdb also relaxed without the ligand, then aligned to the initial one to add the ligand before giving it to ligMPNN
"""



"""
test: 
/work/lpdi/users/eline/miniconda3/envs/ligandmpnn_env/bin/python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/ligMPNN_light_pocket_design.py --pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_1_20_1_T0.2.pdb --scoring /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/scoring/FUN_scoring.py --redesign_d_cutoff 8.0 --target_positions 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 --nstruct 5
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

from tqdm.notebook import tqdm

from colabdesign.af.alphafold.common import protein
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import residue_constants
import jax
import jax.numpy as jnp
import os


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
#
"""STRIP LIGAND AND RELAX"""
#

def strip_ligands(pdb_str):
    return "\n".join(
        line for line in pdb_str.splitlines()
        if (line.startswith("HETATM"))
    ) + "\n"

def str_ligands(pdb_str):
    return "\n".join(
        line for line in pdb_str.splitlines()
        if (line.startswith("ATOM"))
    ) + "\n"

from alphafold.relax import relax
from alphafold.relax import utils
from alphafold.common import protein as p_cf

from colabdesign.af.alphafold.common import protein
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import residue_constants

import pdbfixer
import numpy as np
import jax

        
MODRES = {'MSE':'MET','MLY':'LYS','FME':'MET','HYP':'PRO',
          'TPO':'THR','CSO':'CYS','SEP':'SER','M3L':'LYS',
          'HSK':'HIS','SAC':'SER','PCA':'GLU','DAL':'ALA',
          'CME':'CYS','CSD':'CYS','OCS':'CYS','DPR':'PRO',
          'B3K':'LYS','ALY':'LYS','YCM':'CYS','MLZ':'LYS',
          '4BF':'TYR','KCX':'LYS','B3E':'GLU','B3D':'ASP',
          'HZP':'PRO','CSX':'CYS','BAL':'ALA','HIC':'HIS',
          'DBZ':'ALA','DCY':'CYS','DVA':'VAL','NLE':'LEU',
          'SMC':'CYS','AGM':'ARG','B3A':'ALA','DAS':'ASP',
          'DLY':'LYS','DSN':'SER','DTH':'THR','GL3':'GLY',
          'HY3':'PRO','LLP':'LYS','MGN':'GLN','MHS':'HIS',
          'TRQ':'TRP','B3Y':'TYR','PHI':'PHE','PTR':'TYR',
          'TYS':'TYR','IAS':'ASP','GPL':'LYS','KYN':'TRP',
          'CSD':'CYS','SEC':'CYS'}

def pdb_to_string(pdb_file, chains=None, models=[1]):
  '''read pdb file and return as string'''

  if chains is not None:
    if "," in chains: chains = chains.split(",")
    if not isinstance(chains,list): chains = [chains]
  if models is not None:
    if not isinstance(models,list): models = [models]

  modres = {**MODRES}
  lines = []
  seen = []
  model = 1
  for line in open(pdb_file,"rb"):
    line = line.decode("utf-8","ignore").rstrip()
    if line[:5] == "MODEL":
      model = int(line[5:])
    if models is None or model in models:
      if line[:6] == "MODRES":
        k = line[12:15]
        v = line[24:27]
        if k not in modres and v in residue_constants.restype_3to1:
          modres[k] = v
      if line[:6] == "HETATM":
        k = line[17:20]
        if k in modres:
          line = "ATOM  "+line[6:17]+modres[k]+line[20:]
      if line[:4] == "ATOM":
        chain = line[21:22]
        if chains is None or chain in chains:
          atom = line[12:12+4].strip()
          resi = line[17:17+3]
          resn = line[22:22+5].strip()
          if resn[-1].isalpha(): # alternative atom
            resn = resn[:-1]
            line = line[:26]+" "+line[27:]
          key = f"{model}_{chain}_{resn}_{resi}_{atom}"
          if key not in seen: # skip alternative placements
            lines.append(line)
            seen.append(key)
      if line[:5] == "MODEL" or line[:3] == "TER" or line[:6] == "ENDMDL":
        lines.append(line)
  return "\n".join(lines)


from Bio.PDB import PDBParser, PDBIO, Superimposer, is_aa
from io import StringIO

def align_pdbs_from_strings(reference_pdb_str,
                            align_pdb_str,
                            reference_chain_id,
                            align_chain_id):
    """
    Aligns align_pdb_str onto reference_pdb_str using CA atoms of the
    specified chains. Returns an aligned PDB string (does NOT write files).
    """

    reference_chain_id = reference_chain_id.split(',')[0].strip()
    align_chain_id = align_chain_id.split(',')[0].strip()

    parser = PDBParser(QUIET=True)

    ref_struct = parser.get_structure("ref", StringIO(reference_pdb_str))
    mov_struct = parser.get_structure("mov", StringIO(align_pdb_str))

    ref_model = next(ref_struct.get_models())
    mov_model = next(mov_struct.get_models())

    # Fetch chains
    try:
        ref_chain = ref_model[reference_chain_id]
    except KeyError:
        raise ValueError(f"Reference chain '{reference_chain_id}' not found.")
    try:
        mov_chain = mov_model[align_chain_id]
    except KeyError:
        raise ValueError(f"Align chain '{align_chain_id}' not found.")

    # Build CA maps
    def chain_ca_map(chain):
        ca_map = {}
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            if "CA" in res:
                res_id = res.get_id()
                ca_map[(res_id[1], res_id[2])] = res["CA"]
        return ca_map

    ref_ca = chain_ca_map(ref_chain)
    mov_ca = chain_ca_map(mov_chain)

    common_keys = sorted(
        set(ref_ca.keys()).intersection(mov_ca.keys()),
        key=lambda k: (k[0], k[1] or " ")
    )

    if len(common_keys) < 3:
        raise ValueError(
            f"Not enough matching CA positions to compute superposition ({len(common_keys)} found)"
        )

    fixed_atoms = [ref_ca[k] for k in common_keys]
    moving_atoms = [mov_ca[k] for k in common_keys]

    # Superimpose
    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    rotation, translation = sup.rotran

    # Apply transform to ALL atoms
    for atom in mov_struct.get_atoms():
        atom.transform(rotation, translation)

    # Export structure to a string
    output = StringIO()
    io = PDBIO()
    io.set_structure(mov_struct)
    io.save(output)

    return output.getvalue()


def relax_me(pdb_in, pdb_out): # remove ligand, apply relaxation, put back ligand
  #takes an input pdb write one after modification. 
  # also outputs the relaxed and realigned pdb str
  pdb_str = pdb_to_string(pdb_in)
  ligand_str = str_ligands(pdb_str)
  pdb_str_clean = strip_ligands(pdb_str)
  protein_obj = p_cf.from_pdb_string(pdb_str)
  amber_relaxer = relax.AmberRelaxation(
    max_iterations=0,
    tolerance=2.39,
    stiffness=10.0,
    exclude_residues=[],
    max_outer_iterations=3,
    use_gpu=True)
  relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=protein_obj)
  # align the relaxed pdb back into the original one
  relaxed_aligned_pdb_lines=align_pdbs(reference_pdb_str=pdb_str_clean, align_pdb_str=relaxed_pdb_lines, reference_chain_id="A", align_chain_id="A")
  with open(pdb_out, 'w') as f:
      f.write(relaxed_aligned_pdb_lines)
      f.write("\n")
      f.write(ligand_str)
      f.write("\n END")
  return(relaxed_aligned_pdb_lines + "\n" + ligand_str + "\n END")
        
        
aa_order = residue_constants.restype_order
order_aa = {b:a for a,b in aa_order.items()}





print("Setting up MPNN API")
mpnnrunner = MPNNRunner(model_type="ligand_mpnn", ligand_mpnn_use_side_chain_context=True)  # starting with default checkpoint

# running the relaxation +...
pdb_name = os.path.basename(INPUT_PDB).replace(".pdb", "")
relaxed_pdb_str=relax_me(INPUT_PDB, f"{pdb_name}relaxed.pdb")

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
    #pdbstr = pyrosetta.distributed.io.to_pdbstring(_pose2)
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
            #########################################################
            ### Running MPNN ####
            #########################################################
            # Setting up MPNN runner 
            inp = mpnnrunner.MPNN_Input()
            inp.pdb = relaxed_pdb_str
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
