import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import softmax

from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, Superimposer, MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import StructureBuilder
from Bio.PDB import PDBParser, Selection

"""test
python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/af2/repredict_w_templateBC.py --complex_pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/*... --outdir """

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--complex_pdb', type=str,  nargs="+", required=True, help=' complex pdb file list to clean and use as a reference')
parser.add_argument('--outdir', type=str, required=True, help=' output directory for the cleaned pdb')
args = parser.parse_args()



def copy_structure_with_only_chain(structure, chain_id):
  """
	From BindCraft's Biopyhton_utils : _copy_structure_with_only_chain (https://github.com/martinpacesa/BindCraft) 
	Return a new Structure containing only model 0 and a deep copy of chain `chain_id`."""
  # Build a tiny structure hierarchy: Structure -> Model(0) -> Chain(chain_id) -> Residues/Atoms

  sb = StructureBuilder.StructureBuilder()
  sb.init_structure("single")
  sb.init_model(1)
  sb.init_chain(chain_id)
  # Set segment ID, padded to 4 characters
  sb.init_seg(chain_id.ljust(4))  
  model0 = next(structure.get_models())
  if chain_id not in [c.id for c in model0.get_chains()]:
      raise ValueError(f"Chain '{chain_id}' not found.")
  chain = model0[chain_id]
  for res in chain:
      # Keep only amino-acid residues
      # Assuming is_aa is defined elsewhere and available
      if not is_aa(res, standard=False):
          continue
      hetflag, resseq, icode = res.id
      sb.init_residue(res.resname, hetflag, resseq, icode)

      for atom in res:
          sb.init_atom(atom.name, atom.coord, atom.bfactor, atom.occupancy,
                       atom.altloc, atom.fullname, element=atom.element)
  return sb.get_structure()

# change a chain's id:
def change_chain_id(structure,model_id, old_id, new_id, old_resname=None, new_resname=None):
  chain=structure[model_id][old_id]
  chain.id = new_id
  if new_resname is not None and old_resname is not None:
    for res in structure.get_residues():
      if res.resname.strip().startswith(old_resname):
        res.resname = new_resname
  return(structure)

def rm_ligands(pdb_str):
    return "\n".join(
        line for line in pdb_str.splitlines()
        if (line.startswith("ATOM"))
    ) + "\n"


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



#---------------------------------------------
outdir=args.outdir
complex_pdbs=args.complex_pdb # needs to be split in order to separate binder from target (change chain label for binder residues in pMPNN output)

for complex_pdb in complex_pdbs:
	design_name=os.path.basename(complex_pdb).replace(".pdb", "")
	# binder residues positions:
	from Bio.PDB import PDBParser
	parser = PDBParser(QUIET=True)
	structure = parser.get_structure("x", complex_pdb)
	model = structure[0]             # first model
	chain = model["A"]               # chain A
	# count only standard residues
	from Bio.PDB.Polypeptide import is_aa
	three_to_one = {
	    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
	    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
	    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
	    # common variants
	    "MSE":"M",  # Selenomethionine
	}
	residues = [res for res in chain.get_residues() if is_aa(res, standard=True)]
	#print("residues:", residues)
	print("len residues:", len(residues))
	# extract/trim the binder sequence
	binder_length=len(residues)-256
	print("binder len:", binder_length)
	
	from Bio.PDB import PDBIO, Chain
	from Bio.PDB.Polypeptide import is_aa
	# change chain id for binder residues (from A to B) and for ligand atoms: from B to L 
	# ligand:
	structure=change_chain_id(structure=structure,model_id=0, old_id="B", new_id="L", old_resname=None, new_resname=None)
	#binder:
	for model in structure:
	    # Retrieve chain A (binder assumed to be first part)
	    model = structure[0]             # first model
	    chain_A = model["A"] 
	    lig_chain=model["B"]
	    
	    residues_A = list(chain_A.get_residues())
	    #print("res A:", residues_A)
	    if binder_length > len(residues_A):
	        raise ValueError("binder_length exceeds number of residues in chain A")
	    # Create new chain B
	    chain_B = Chain.Chain("B")
	    count=0
	    # Transfer the binder residues into chain B
	    for i, residue in enumerate(residues_A):
	        if not is_aa(residue, standard=False):
	            continue
	        if i < binder_length:
	            # Remove from chain A
	            chain_A.detach_child(residue.id)
	            # Add to chain B
	            chain_B.add(residue)
	            count+=1
	    # Add new chain B to the model
	    model.add(chain_B)
	    print("len chain B:",count)
	
	
	# Save modified structure
	path=os.path.join(outdir, design_name + ".pdb")
	
	io = PDBIO()
	io.set_structure(structure)
	io.save(path)

print("Done cleaning pdbs")
