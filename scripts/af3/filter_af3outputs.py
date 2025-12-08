import pandas as pd
from Bio.PDB.Polypeptide import is_aa
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
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss, get_plddt_loss, get_exp_res_loss, get_pae_loss, get_con_loss, get_rmsd_loss, get_dgram_loss, get_fape_loss
from colabdesign.shared.utils import copy_dict
from colabdesign.shared.prep import prep_pos
from Bio.PDB import PDBIO, StructureBuilder, PDBParser
### Path to this cloned GitHub repo:
SCRIPT_DIR = "/work/lpdi/users/eline/smol_binder_diffusion_pipeline"  # edit this to the GitHub repo path. Throws an error by default.
assert os.path.exists(SCRIPT_DIR)
sys.path.append(SCRIPT_DIR + "/scripts/utils")
import utils


#----------------------------------------------------------------------------------------------------------
"""-----------------------------------------------------SETUP-----------------------------------------------------"""
#----------------------------------------------------------------------------------------------------------

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
AF3_DIR = f"{WDIR}/4_af3"

os.chdir(AF3_DIR)



df1=pd.read_csv("output/design_confidences.csv")
df2=pd.read_csv("output2/design_confidences.csv")
df=pd.concat([df1,df2], ignore_index=True)
"""
# ensure unique ids
# count occurrences per group
counts = df.groupby("id").cumcount()

# for duplicates (count > 0), modify the id
df["id"] = df["id"].replace("seq0","seq") + counts.astype(str)
"""
# split col chain_plddt into binder_plddt, ligand_plddt
import ast

df["chain_plddt"] = df["chain_plddt"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)
df["binder_plddt"] = df["chain_plddt"].apply(lambda d: d.get("B"))
df["ligand_plddt"] = df["chain_plddt"].apply(lambda d: d.get("FUN"))

# plotting our metrics: iptm, full_rmsd, binder_plddt, ligand_plddt
# scatter plot for case n°1
_=sns.scatterplot(data=df, x='full_rmsd', y='iptm')
full_rmsd = 2.0
plt.axvline(x=full_rmsd, ymin=0, ymax=1, color="black", linestyle="--")
iptm = 0.8
plt.axhline(
    y=iptm, xmin=0, xmax=1, color="black", linestyle="--"
)
plt.xlabel("full_rmsd ")
plt.ylabel("i_pTM")
plt.title("rmsd vs iptm")
plt.legend(title="Binders selected with case 1 filters")
plt.savefig(f"rmsd_vs_iptm.png")
plt.close()



# filter
# plot final selected binders 
selected1_binder_list=df[(df['full_rmsd'] <=1.5) & (df['iptm'] >=0.8)& (df['binder_rmsd']<=1.5)
& (df['binder_plddt'] >=0.8) & (df['ligand_plddt'] >=0.85)].id
df['selected1'] = df['id'].isin(selected1_binder_list)

success_df=df[(df['full_rmsd'] <=1.5) & (df['iptm'] >=0.8)& (df['binder_rmsd']<=1.5)
& (df['binder_plddt'] >=0.8) & (df['ligand_plddt'] >=0.85)]
# write out successful binders .csv file
success_df.to_csv(f"selected1_binder_metrics_df.csv", index=False)

succes_df=pd.read_csv(f"selected1_binder_metrics_df.csv")
#for those binders, repredict the complex target + binder without ligand and select only those with a low ipTM:
# reprediction with colabDEsign form target template + binder sequence

# target template: 

def _copy_structure_with_only_chain(structure, chain_id):
    """Return a new Structure containing only model 0 and a deep copy of chain `chain_id`."""
    # Build a tiny structure hierarchy: Structure -> Model(0) -> Chain(chain_id) -> Residues/Atoms

    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("single")
    sb.init_model(1)
    sb.init_chain(chain_id)
    # Set segment ID, padded to 4 characters
    sb.init_seg(chain_id.ljust(4))    
    model0 = structure[0]
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

def extract_chain(input_cif_path: str, template: str, chain_id: str):
    """
    Extracts a specific chain from a PDB file using _copy_structure_with_only_chain
    and saves it to a new PDB file with explicit MODEL/ENDMDL records.

    Args:
        input_pdb_path (str): Path to the input PDB file (complex).
        output_pdb_path (str): Path to save the extracted chain PDB file.
        chain_id (str): The identifier of the chain to extract (e.g., "A", "B").
    """
    parser = MMCIFParser()
    structure = parser.get_structure("protein", input_cif_path)
    io = PDBIO(use_model_flag=1)

    # Use the helper function to get a new structure with only the desired chain
    new_structure = _copy_structure_with_only_chain(structure, chain_id)
    # Save the new structure, explicitly writing model records
    io.set_structure(new_structure)
    io.save(template)


def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    unaligned RMSD of binder compared to original trajectory, in other words how far is binder in the repredicted complex from the original binding site
    Compute Cα RMSD between chains in two PDBs *without* superposition.
    Residues are matched by (resseq, insertion code) intersection.

    Parameters
    ----------
    reference_pdb : str
        Path to the reference PDB file.
    align_pdb : str
        Path to the PDB file to compare against.
    reference_chain_id : str
        Chain ID in the reference structure; if comma-separated, only the first is used.
    align_chain_id : str
        Chain ID in the moving structure; if comma-separated, only the first is used.

    Returns
    -------
    float
        RMSD in Å, rounded to 2 decimals.

    Raises
    ------
    ValueError
        If chains are missing or there are fewer than 3 matched residues with Cα atoms.
    """
    # Use first value if comma-separated
    reference_chain_id = reference_chain_id.split(',')[0].strip()
    align_chain_id = align_chain_id.split(',')[0].strip()

    pdb_parser = PDBParser(QUIET=True)
    cif_parser = MMCIFParser (QUIET=True)
    ref_struct = cif_parser.get_structure("ref", reference_pdb)
    mov_struct = pdb_parser.get_structure("mov", align_pdb)

    ref_model = next(ref_struct.get_models())
    mov_model = next(mov_struct.get_models())

    # Fetch chains
    try:
        ref_chain = ref_model[reference_chain_id]
    except KeyError:
        raise ValueError(f"Reference chain '{reference_chain_id}' not found in {reference_pdb}.")
    try:
        mov_chain = mov_model[align_chain_id]
    except KeyError:
        raise ValueError(f"Align chain '{align_chain_id}' not found in {align_pdb}.")

    # Build maps of Cα atoms keyed by (resseq, icode)
    def ca_map(chain):
        out = {}
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            if "CA" in res:
                hetflag, resseq, icode = res.get_id()
                out[(resseq, icode)] = res["CA"]
        return out

    ref_ca = ca_map(ref_chain)
    mov_ca = ca_map(mov_chain)

    # Common residue keys, ordered by residue number then insertion code
    common = sorted(set(ref_ca.keys()).intersection(mov_ca.keys()),
                    key=lambda k: (k[0], (k[1] or " ")))

    if len(common) < 3:
        raise ValueError(
            f"Not enough matched residues with Cα to compute RMSD without alignment "
            f"(found {len(common)})."
        )

    ref_coords = np.array([ref_ca[k].get_coord() for k in common], dtype=float)
    mov_coords = np.array([mov_ca[k].get_coord() for k in common], dtype=float)

    # Unaligned RMSD = sqrt(mean(||ref - mov||^2))
    diffs = ref_coords - mov_coords
    rmsd = float(np.sqrt((diffs * diffs).sum(axis=1).mean()))
    return round(rmsd, 2)


# and a function to extract relevant binders from each final_stat.csv df from bindcraft
def extract_filtered_binders(filtered_df, source, dest):
  """ 
  copies the interesting binders in filtered_df from accepted_folder to filtered_binders_pdbs
  source is the col name in the df where the structure path is stored for each binder
  """
  print(f"Filters on specific reprediction scores gave {len(filtered_df['id'])} binders: {filtered_df['id']}")

  str_binders=''
  for pdb_path in filtered_df[source]:
    #filename = os.path.basename(pdb_path)
    #out_path = os.path.join(dest, filename)
    #name, ext = os.path.splitext(filename)
    """if os.path.exists(outpath):
        newname=name+
        newpath=os.path.joint(dest,newname)"""
    id_index=filtered_df['complex_path']==pdb_path
    name=filtered_df.loc[id_index, "id"]+".cif"
    
    str_binders+=f"{pdb_path} "
    # also copy the relevant pdbs to a filtered_binders folder:
    shutil.copy(pdb_path, os.path.join(dest,name))

  print(str_binders)

#----------------------------------------------------------------------------------
from Bio.PDB import MMCIFParser
TEMP_DIR = f"{AF3_DIR}/reprediction_templates"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR, exist_ok=True)

for design_name in success_df.id:
  if "seq0" in design_name:
    name=design_name.replace("_seq0","")
    cif_path=f"{AF3_DIR}/output/{name}/{name}_model.cif"

  if "seq1" in design_name:
    name=design_name.replace("_seq1","")
    cif_path=f"{AF3_DIR}/output2/{name}/{name}_model.cif"
  clear_mem()
  params = '/work/lpdi/users/goldbach/software/colabdesign/params' 
  
  # extract target template:
  template=f"{TEMP_DIR}/{design_name}.pdb"
  extract_chain(input_cif_path=cif_path, template=template, chain_id="A") # extract the target structure from the AF3 model (chain A)
  # extract binder sequence from the AF3 model (chain B)
  parser = MMCIFParser()
  structure = parser.get_structure("protein", cif_path)
  binder_chain=structure[0]["B"]
  from Bio.PDB.Polypeptide import is_aa
  three_to_one = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
    # common variants
    "MSE":"M",  # Selenomethionine
  }
  residues = [res for res in binder_chain.get_residues() if is_aa(res, standard=True)]
  binder_length=len(residues)
  res_letters = []
  for res in residues:
      try:
          aa = three_to_one[res.resname]
          res_letters.append(aa)
      except KeyError:
          raise ValueError(f"Unknown residue: {res.resname} at {res.id}")
  
  binder_sequence = "".join(res_letters)
  print(binder_sequence)
  #run reprediction with Colab Design:
  # compile complex prediction model
  model = mk_afdesign_model(protocol="binder", num_recycles=3, data_dir=params, 
                                              use_multimer=True,
                                              use_templates=True,
                                               use_initial_guess=False, #Introduce bias by providing binder atom positions as a starting point for prediction.
                                               use_initial_atom_pos=False) # Introduce atom position bias into the structure module for atom initilisation.
  
  
  model.prep_inputs(pdb_filename=template,
                        chain="A",
                        #binder_chain="B",# do not specifiy if the template only contains the target
                        binder_len=binder_length,
                        rm_target_seq=False, #b
                        use_binder_template=False #a
                        #,rm_template_ic=True #c
                        )
  
  prediction_stats = {}
  REP_DIR = f"{AF3_DIR}/no_lig_repredictions"
  if not os.path.exists(REP_DIR):
      os.makedirs(REP_DIR, exist_ok=True)
  for model_num in [0,1]:
    model.predict(seq=binder_sequence,
                  models=[model_num],
                  num_recycles=3)

    predicted_complex_pdb = os.path.join(REP_DIR, f"{design_name}_model_{model_num+1}_repredicted_nolig.pdb")
    model.save_pdb(predicted_complex_pdb)
    prediction_metrics = copy_dict(model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

    # extract the statistics for the model
    stats = {
        f"nolig_pLDDT": round(prediction_metrics['plddt'], 2),
        f"nolig_pTM": round(prediction_metrics['ptm'], 2),
        f"nolig_i_pTM": round(prediction_metrics['i_ptm'], 2),
        f"nolig_pAE": round(prediction_metrics['pae'], 2),
        f"nolig_i_pAE": round(prediction_metrics['i_pae'], 2)
          }


    # unaligned RMSD calculate to determine if binder is in the designed binding site
    rmsd_site = unaligned_rmsd(cif_path, predicted_complex_pdb, "B", "B")
    stats[f"nolig_binder_rmsd"] = rmsd_site # this should be used to filter the models that are binding in the predicted binding site

    prediction_stats[model_num+1] = stats # 2 dictionnaries index 1 and 2 to eventually add to the metrics df

  data={}
  for key in prediction_stats[1].keys():
    data[key]=(prediction_stats[1][key] + prediction_stats[2][key])/2
  data["id"]=design_name
  data["binder_sequence"]=binder_sequence
  data["complex_path"]=cif_path

  df = pd.DataFrame([data])
  df.to_csv(f"{REP_DIR}/no_lig_confidences.csv", mode="a", index=False, header=not pd.io.common.file_exists(f"{REP_DIR}/no_lig_confidences.csv"))

#----------------------------------------------------------------------------------------------------------------------------------------
REP_DIR = f"{AF3_DIR}/no_lig_repredictions"
df=pd.read_csv(f"{REP_DIR}/no_lig_confidences.csv")
success_df=pd.read_csv(f"selected1_binder_metrics_df.csv")
# at the end: merge the two confidence dfs:
df_merged = pd.merge(success_df, df, on="id", how="inner")
df_merged.to_csv(f"{AF3_DIR}/design_all_confidences.csv")

import seaborn as sns
# scatter plot for iptm
_=sns.scatterplot(data=df_merged, x='nolig_i_pTM', y='iptm')
nolig_i_pTM = 0.5
plt.axvline(x=nolig_i_pTM, ymin=0, ymax=1, color="black", linestyle="--")
iptm = 0.8
plt.axhline(
    y=iptm, xmin=0, xmax=1, color="black", linestyle="--"
)
plt.xlabel("nolig_i_pTM ")
plt.ylabel("i_pTM")
plt.title("nolig_i_pTM vs iptm")
plt.legend(title="nolig_i_pTM vs iptM with ligand")
plt.savefig(f"nolig_i_pTM_vs_iptm.png")
plt.close()

# scatter plot for binder rmsd
_=sns.scatterplot(data=df_merged, x='nolig_binder_rmsd', y='binder_rmsd')
nolig_binder_rmsd = 3
plt.axvline(x=nolig_binder_rmsd, ymin=0, ymax=1, color="black", linestyle="--")
binder_rmsd = 2.0
plt.axhline(
    y=binder_rmsd, xmin=0, xmax=1, color="black", linestyle="--"
)
plt.xlabel("nolig_binder_rmsd ")
plt.ylabel("binder_rmsd")
plt.title("nolig_binder_rmsd vs binder_rmsd")
plt.legend(title="nolig_binder_rmsd vs binder_rmsd with ligand")
plt.savefig(f"nolig_binder_rmsd_vs_binder_rmsd.png")
plt.close()

# filter
# plot final selected binders 
selected2_binder_list=df_merged[(df_merged['nolig_binder_rmsd'] >=1.5) & (df_merged['nolig_i_pTM'] <=0.5)& (df_merged['nolig_pLDDT']>=0.7)].id
df_merged['selected2'] = df_merged['id'].isin(selected2_binder_list)

success_df2=df_merged[(df_merged['nolig_binder_rmsd'] >=1.5) & (df_merged['nolig_i_pTM'] <=0.5)& (df_merged['nolig_pLDDT']>=0.7)]
# write out successful binders .csv file
filtered=f"{AF3_DIR}/filtered"

success_df2.to_csv(f"{filtered}/selected2_binder_all_metrics_df.csv", index=False)

extract_filtered_binders(success_df2, "complex_path", filtered)

