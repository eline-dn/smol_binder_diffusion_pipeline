import sys, os, glob, shutil, subprocess
import json
import getpass
import argparse
import random
import copy
import time
import scipy.spatial
import io
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import os

from Bio.PDB import PDBParser, PDBIO, Superimposer, is_aa
from io import StringIO


SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_DIR}/../../lib/alphafold")

from alphafold.relax import relax
from alphafold.relax import utils
from alphafold.common import protein as p_cf

import pdbfixer
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True, type=str, help="Input PDB")
parser.add_argument("--backbone_pdb", required=True, type=str, help="pMPNN output backbone, single chain and with ligand PDB")
args = parser.parse_args()
INPUT_PDB = args.pdb
BB_PDB=args.backbone_pdb



#
"""STRIP LIGAND AND RELAX"""
#

def rm_ligands(pdb_str):
    return "\n".join(
        line for line in pdb_str.splitlines()
        if (line.startswith("ATOM"))
    ) + "\n"

def str_ligands(pdb_str):
    return "\n".join(
        line for line in pdb_str.splitlines()
        if (line.startswith("HETATM"))
    ) + "\n"

def to_one_chain_nope(pdb_str, chain_id="A"): # doesn't work
    lines=list()
    for line in pdb_str.splitlines():
        if (line.startswith("ATOM")):
            chain = line[21:22]
            if chain != chain_id:
                line[21:22]=chain_id
        lines.append(line)
    return("\n".join(lines))   

def to_one_chain(pdb_str, chain_id="A"):
    new_lines = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM"):
            # chain ID is in column 22 → index 21
            if len(line) >= 22:
                # rebuild the line with modified chain ID
                line = line[:21] + chain_id + line[22:]
        new_lines.append(line)
    return "\n".join(new_lines)

    
       
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


def relax_me(pdb_in, pdb_out, ligand_str, bb_pdb_str): #  apply relaxation, align, put back ligand
  #takes an input pdb write one after modification. 
  # also outputs the relaxed and realigned pdb str
  bb_pdb_str_clean=rm_ligands(bb_pdb_str)
  pdb_str = pdb_to_string(pdb_in)
  #ligand_str = str_ligands(pdb_str)
  pdb_str_clean = to_one_chain(pdb_str, chain_id="A")
  print(pdb_str_clean)
  protein_obj = p_cf.from_pdb_string(pdb_str_clean)
  amber_relaxer = relax.AmberRelaxation(
    max_iterations=0,
    tolerance=2.39,
    stiffness=10.0,
    exclude_residues=[],
    max_outer_iterations=3,
    use_gpu=True)
  relaxed_pdb_lines, _, _ = amber_relaxer.process(prot=protein_obj)
  # align the relaxed pdb back into the mpnn bb one
  relaxed_aligned_pdb_lines=align_pdbs(reference_pdb_str=bb_pdb_str_clean, align_pdb_str=relaxed_pdb_lines, reference_chain_id="A", align_chain_id="A")
  with open(pdb_out, 'w') as f:
      f.write(relaxed_aligned_pdb_lines)
      f.write("\n")
      f.write(ligand_str)
      f.write("\n END")
  return(relaxed_aligned_pdb_lines + "\n" + ligand_str + "\n END")

#-------------------------------------------------------------------------------------------------
"""
change repredicted PDB in order to have it as one chain
relax it
realign it to the pMPNN backbone
put ligand back in
save for lig MPNN pocket redesign
"""

# running the relaxation 
bb_pdb_str = pdb_to_string(BB_PDB)
ligand_str = str_ligands(bb_pdb_str)
pdb_name = os.path.basename(INPUT_PDB).replace(".pdb", "")
relaxed_pdb_str=relax_me(INPUT_PDB, f"{pdb_name}relaxed.pdb", ligand_str, bb_pdb_str) # also saves the relaxed aligned + ligand in pdb out
print( f"Saved {pdb_name}relaxed.pdb")
