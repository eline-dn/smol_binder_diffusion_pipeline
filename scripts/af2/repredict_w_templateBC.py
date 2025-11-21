import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss, get_plddt_loss, get_exp_res_loss, get_pae_loss, get_con_loss, get_rmsd_loss, get_dgram_loss, get_fape_loss
from colabdesign.shared.utils import copy_dict
from colabdesign.shared.prep import prep_pos
"""test
python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/af2/repredict_w_templateBC.py --complex_pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_1_20_1_T0.2.pdb --scorefile score.txt
"""

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--complex_pdb', type=str, required=True, help=' complex pdb file name to extract the sequence from and use as a template')
#parser.add_argument('--af-models',nargs='+', default="4", help='AlphaFold models to run (1-5)')
#parser.add_argument('--af-nrecycles', type=int, default=3, help='Number of recycling iterations for AlphaFold')
parser.add_argument('--scorefile', type=str, default="scores.csv", help='Scorefile name. (default = scores.csv)')

args = parser.parse_args()


def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, prediction_models,  design_path, seed=None):
    prediction_stats = {}

    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())



    # start prediction per AF2 model, 2 are used by default due to masked templates
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        complex_pdb = os.path.join(design_path, f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(complex_pdb):
            # predict model
            prediction_model.predict(seq=binder_sequence, models=[model_num], num_recycles=3, verbose=False)
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'i_pTM': round(prediction_metrics['i_ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2), 
                'i_pAE': round(prediction_metrics['i_pae'], 2)
            }
            prediction_stats[model_num+1] = stats

    return prediction_stats

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

#---------------------------------------------
clear_mem()
params = '/work/lpdi/users/goldbach/software/colabdesign/params' 
complex_pdb=args.complex_pdb # needs to be split in order to separate binder from target (change chain label for binder residues in pMPNN output)
design_name=os.path.basename(complex_pdb)

# remove the ligand from the pdb:
pdb_str = pdb_to_string(complex_pdb)
pdb_str_clean = rm_ligands(pdb_str)
path=complex_pdb.replace(".pdb","")
complex_pdb_clean=f"{path}_nolig.pdb"
with open(complex_pdb_clean, 'w') as f:
      f.write(pdb_str_clean)

# binder residues positions:
from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)
structure = parser.get_structure("x", complex_pdb_clean)
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
#binder_sequence="".join(residues[:binder_length+1])
# convert residues to one-letter sequence
res_letters = []
for res in residues[:binder_length]:
    try:
        aa = three_to_one[res.resname]
        res_letters.append(aa)
    except KeyError:
        raise ValueError(f"Unknown residue: {res.resname} at {res.id}")

binder_sequence = "".join(res_letters)
print(binder_sequence)
print("len binder seq:", len(binder_sequence))
# change chain id for binder residues (from A to B):
from Bio.PDB import PDBIO, Chain

# change chain id for binder residues (from A to B):
for model in structure:
    # Retrieve chain A (binder assumed to be first part)
    model = structure[0]             # first model
    chain_A = model["A"] 
    residues_A = list(chain_A.get_residues())
    print("res A:", residues_A)
    if binder_length > len(residues_A):
        raise ValueError("binder_length exceeds number of residues in chain A")
    # Create new chain B
    chain_B = Chain.Chain("B")
    count=0
    # Transfer the binder residues into chain B
    for i, residue in enumerate(residues_A):
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
path=complex_pdb_clean.replace('.pdb','')
complex_pdb_clean_split = f"{path}_nolig_split.pdb"

io = PDBIO()
io.set_structure(structure)
io.save(complex_pdb_clean_split)

design_paths="./"
# compile complex prediction model
complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=3, data_dir=params, 
                                            use_multimer=True,
                                             use_initial_guess=False, #Introduce bias by providing binder atom positions as a starting point for prediction.
                                             use_initial_atom_pos=False) # Introduce atom position bias into the structure module for atom initilisation.

complex_prediction_model.prep_inputs(pdb_filename=complex_pdb_clean_split,
                                         chain='A',
                                         binder_chain='B',
                                         binder_len=binder_length,
                                         use_binder_template=True,
                                         rm_target_seq=False, #remove target template sequence for reprediction (increases target flexibility)
                                        rm_template_ic=True)


mpnn_complex_statistics = predict_binder_complex(prediction_model=complex_prediction_model,
                                                                    binder_sequence=binder_sequence, 
                                                                    mpnn_design_name=design_name,
                                                                   prediction_models=[1],
                                                                    design_path= design_paths)

print(f"Predicted complex structure for template {complex_pdb} \n Saved in {design_paths}")
