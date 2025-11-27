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

DESIGN_DIR_ligMPNNoutput= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/ligMPNN_output"
os.chdir(DESIGN_DIR_ligMPNNoutput)

"""---------- co folding with af 3---------"""
"""
### prep the json input files
import json

# path to your input JSON
input_path = f"{AF3_DIR}/input2/test_1Z9Y_FUN_no_msa_template.json" # template generic json file to use as input for af3


# load JSON
with open(input_path, "r") as f:
    data = json.load(f)

### First collecting MPNN outputs and creating FASTA files for AF2 input
mpnn_fasta = utils.parse_fasta_files(glob.glob(f"{DESIGN_DIR_ligMPNNoutput}/*.fasta"))

mpnn_fasta_clean_half={}
slot=1
count=0
num_seq=len(mpnn_fasta.keys())

for id, seq in mpnn_fasta.items():
  count+=1
  if count > num_seq/4:
    count=0
    slot+=1
  str_slot=str(slot)
  json_dir=f"{AF3_DIR}/input2/{str_slot}"
  if not os.path.exists(json_dir):
    os.makedirs(json_dir, exist_ok=True)
  if "seq0" in id:
    continue # keep only seq0 right now and see if we already have binders # changed later to keep only seq 1 and repredict the rest
  id_clean=id.replace(">","")
  id_clean=id_clean.replace("_seq1\n","")
  id_clean=id_clean.replace("model2_w_ligand_fused_","")
  seq_clean=seq.replace("\n","")
  seq_clean=seq_clean[:-256]
  #mpnn_fasta_clean_half[id_clean]=seq_clean[:-256]
  # modify protein B sequence
  output_path = f"{json_dir}/{id_clean}.json" # binder sequence specific json input for each af3 
  data["name"]=id_clean
  for entry in data["sequences"]:
      if "protein" in entry:
          if entry["protein"]["id"] == ["B"]:
              entry["protein"]["sequence"] = seq_clean
  
  # write modified JSON to a new file
  with open(output_path, "w") as f:
      json.dump(data, f, indent=2)
  
"""

"""
sbatch /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/af3/run_alphafold.sh -i input2/1 -o output2 --no-msa
sbatch /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/af3/run_alphafold.sh -i input2/2 -o output2 --no-msa
sbatch /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/af3/run_alphafold.sh -i input2/3 -o output2 --no-msa
sbatch /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/af3/run_alphafold.sh -i input2/4 -o output2 --no-msa


sbatch run_alphafold.sh -i /work/lpdi/users/dobbelst/tools/alphafold3_examples/af_input/fold_input_singleseq.json -o <OUTPUT_DIR> --no-msa
"""

#### process & filter output
# in  /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/4_af3/output
"""
filtering strategy: good iptm and plddt, + low rmsd to original RF diffusion backbone
(then repredicted complex without ligand , keep low iptm and plddt +  high rmsd to RF diff backbone)
"""
import json
import re
from Bio.PDB import PDBParser, MMCIFParser, Superimposer, DSSP, Selection, Polypeptide, PDBIO, Select, Chain
from Bio.PDB.Polypeptide import is_aa

AF3_struct= f"{AF3_DIR}/output"
for confidence in glob.glob(f"{AF3_struct}/*/*_summary_confidences.json"):
    design_name=os.path.basename(confidence)
    design_name=design_name.replace("_summary_confidences", "") # e.g. t2_2_7_4_t0.3_ltp0.2_dcut500.0
    #load JSON
    with open(confidence, "r") as f:
        data = json.load(f) # data["chain_iptm", "chain_pair_iptm", "chain_pair_pae_min","chain_ptm","fraction_disordered": 0.0, "has_clash": 0.0, "iptm": 0.7, "ptm": 0.75, "ranking_score": 0.71]
    data["id"]=f"{design_name}_seq0"
    # also get atoms plddts from ..._confidences.json:
    conf=confidence.replace("_summary", "")
    with open(conf,"r") as f:
        conf=json.load(f)
    data["atom_plddt"]=conf["atom_plddts"]
    data["atom_chain_ids"]=conf["atom_chain_ids"]
    data["chain_plddt"]={}
    chain_atom_len={}
    for i,chain_id in enumerate(data["atom_chain_ids"]):
        if chain_id not in data["chain_plddt"].keys():
            data["chain_plddt"][chain_id]=0
        data["chain_plddt"][chain_id]+=data["atom_plddt"][i]
        if chain_id not in chain_atom_len.keys():
            chain_atom_len[chain_id]=0
        chain_atom_len[chain_id]+=1
    for chain, plddt in data["chain_plddt"].items():
        data["chain_plddt"][chain]=plddt/chain_atom_len[chain] # = mean atom plddt per chain
    
    # compute aligned rmsd to original rf diffusion bb:
    # retrieve bb: format t2_1_100_1_T0.3.pdb in MPNN_DIR
    bb_name = re.sub(r"_ltp0\.[1-9]_dcut(8\.0|15\.0|500\.0).json", ".pdb", design_name)
    bb_name= bb_name.replace("t0","T0")
    bb_pdb=f"{MPNN_DIR}/backbones/{bb_name}"
    # load / convert (?) model mmcif
    design_cif=confidence.replace("_summary_confidences.json","_model.cif")
    pdb_parser = PDBParser(QUIET=True)
    mmcif_parser=MMCIFParser(QUIET= True)
    ref_struct = pdb_parser.get_structure("ref", bb_pdb)
    mov_struct = mmcif_parser.get_structure("mov", design_cif)
    
    # before alignment, split binder and target in pMPNN outputs :
    #create new structure  with structure builder:
    # original chain A
    model = ref_struct[0]
    chainA = model["A"]
    chainB=model["B"]
    residues = [res for res in chainA if is_aa(res, standard=False)]
    N = len(residues)

    cutoff_index = N - 256

    # ------ Create new structure with a new Model ------
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain

    new_struct = Structure("split")
    new_model = Model(0)

    chain_C = Chain("C")
    chain_A = Chain("A")

    # ------ Fill chains directly with residue objects ------
    for i, res in enumerate(residues):
        if i < cutoff_index:
            chain_C.add(res.copy())   # copy to avoid pointer alias
        else:
            chain_A.add(res.copy())

    new_model.add(chain_C)
    new_model.add(chain_A)
    new_model.add(chainB) # put back ligand
    new_struct.add(new_model)

            
    # in ref: chain A is target, chB is ligand, chC is binder
    # in mov: chain A is target, chB is binder
    # align chain A to chain A and compute  binder rmsd to original binding site + QC for target rmsd to reference target
    # Use first model (index 0) for both
    ref_model = next(new_struct.get_models())
    mov_model = next(mov_struct.get_models())
    reference_chain_id="A"
    align_chain_id="A"
    # Fetch chains
    try:
        ref_chain = ref_model[reference_chain_id]
    except KeyError:
        raise ValueError(f"Reference chain '{reference_chain_id}' not found in {bb_pdb}.")
    try:
        mov_chain = mov_model[align_chain_id]
    except KeyError:
        raise ValueError(f"Align chain '{align_chain_id}' not found in {design_cif}.")

    # Build resseq -> CA atom maps for standard residues
    def chain_ca_map(chain):
        ca_map = {}
        for res in chain:
            # Skip hetero/water; only standard amino acids
            if not is_aa(res, standard=True):
                continue
            if "CA" in res:
                resseq = res.get_id()[1]  # (hetero flag, resseq, icode) -> take numerical resseq
                # If there are insertion codes, you could include res.get_id()[2] too,
                # but for most cases resseq is enough to match.
                ca_map[(resseq, res.get_id()[2])] = res["CA"]
        return ca_map

    ref_ca = chain_ca_map(ref_chain)
    mov_ca = chain_ca_map(mov_chain)

    # Intersect by (resseq, icode)
    common_keys = sorted(set(ref_ca.keys()).intersection(mov_ca.keys()),
                         key=lambda k: (k[0], (k[1] or " ")))

    if len(common_keys) < 3:
        raise ValueError(
            f"Not enough matching residues between chains {reference_chain_id} (ref) and "
            f"{align_chain_id} (mov) to compute a reliable superposition (found {len(common_keys)})."
        )

    fixed_atoms = [ref_ca[k] for k in common_keys]
    moving_atoms = [mov_ca[k] for k in common_keys]

    # Superimpose
    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    # Apply transform to ALL atoms in the moving structure
    rotation, translation = sup.rotran
    for atom in mov_struct.get_atoms():
        atom.transform(rotation, translation)
    rmsd_target = sup.rms

    data["target_rmsd"]=round(rmsd_target,2)
    print("target rmsd:", rmsd_target)

    # -------compute binder rmsd to original binding site:--------------------------
    ref_binder_chain = ref_model["C"]
    mov_binder_chain = mov_model["B"]
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

    ref_binder_ca = ca_map(ref_binder_chain)
    mov_binder_ca = ca_map(mov_binder_chain)

    # Common residue keys, ordered by residue number then insertion code
    common_binder = sorted(set(ref_binder_ca.keys()).intersection(mov_binder_ca.keys()),
                    key=lambda k: (k[0], (k[1] or " ")))

    if len(common_binder) < 3:
        raise ValueError(
            f"Not enough matched residues with Cα to compute RMSD without alignment "
            f"(found {len(common_binder)})."
        )

    ref_coords_binder = np.array([ref_binder_ca[k].get_coord() for k in common_binder], dtype=float)
    mov_coords_binder = np.array([mov_binder_ca[k].get_coord() for k in common_binder], dtype=float)

    # Unaligned RMSD = sqrt(mean(||ref - mov||^2))
    diffs = ref_coords_binder - mov_coords_binder
    rmsd_binder = float(np.sqrt((diffs * diffs).sum(axis=1).mean()))

    print("binder rmsd:", rmsd_binder)
    data["binder_rmsd"]=round(rmsd_binder, 2)

    # add metrics to csv file
    df = pd.DataFrame([data])
    df.to_csv(f"{AF3_struct}/design_confidences.csv", mode="a", index=False, header=not pd.io.common.file_exists(f"{AF3_struct}/design_confidences.csv"))
    

    

