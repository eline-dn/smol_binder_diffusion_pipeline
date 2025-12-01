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
from Bio.PDB import PDBParser, MMCIFParser, Superimposer, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, PDBIO
from Bio.PDB.Polypeptide import is_aa

AF3_struct= f"{AF3_DIR}/output"
for confidence in glob.glob(f"{AF3_struct}/*/*_summary_confidences.json"):
    print("condidence:", confidence)

    
    design_name=os.path.basename(confidence)
    design_name=design_name.replace("_summary_confidences", "") # e.g. t2_2_7_4_t0.3_ltp0.2_dcut500.0
    design_name=design_name.replace(".json", "")
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
    bb_name = re.sub(r"_ltp0\.[1-9]_dcut(8\.0|15\.0|500\.0)", ".pdb", design_name)
    bb_name= bb_name.replace("t0","T0")
    bb_pdb=f"{MPNN_DIR}/backbones/{bb_name}"
    print("bb:", bb_pdb)
    # load / convert (?) model mmcif
    design_cif=confidence.replace("_summary_confidences.json","_model.cif")
    print("design_cif:", design_cif)
    pdb_parser = PDBParser(QUIET=True)
    mmcif_parser=MMCIFParser(QUIET= True)
    ref_struct = pdb_parser.get_structure("ref", bb_pdb)
    mov_struct = mmcif_parser.get_structure("mov", design_cif)
    print(bb_pdb)
    print(design_cif)
    
    ref_model = ref_struct[0]
    mov_model = mov_struct[0]
    
    from Bio.PDB import PDBParser, MMCIFParser, Superimposer, StructureBuilder
    from Bio.PDB.Polypeptide import is_aa
    # -------------------------------------------------------------
    # 2) Split reference: chain A = fused binder+target
    #    We assume binder comes first and target comes after.
    #    User must provide binder_length (or compute from sequence).
    # -------------------------------------------------------------
    def split_fused_chain(ref_model, fused_chain_id, binder_len):
        fused = ref_model[fused_chain_id]
    
        # new chains
        builder = StructureBuilder.StructureBuilder()
        builder.init_structure("split")
        builder.init_model(0)
        builder.init_chain("C")   # binder
        builder.init_chain("A")   # target
    
        residues = [res for res in fused if is_aa(res, standard=True)]
        #print(residues)
        complex_len=len(residues)
        print("complex_len:", complex_len)
        for i, res in enumerate(residues):
            new_res = res.copy()
            if i < complex_len-binder_len:
                builder.structure[0]["C"].add(new_res)
            else:
                builder.structure[0]["A"].add(new_res)
    
        # also copy ligand (chain B)
        if "B" in ref_model:
            builder.init_chain("F")
            for res in ref_model["B"]:
                new_res = res.copy()
                builder.structure[0]["F"].add(new_res)
        """
        for atom in builder.structure[0]["C"].get_atoms():
            print(atom.get_full_id())
        """
        return builder.get_structure()
    
    
    # You must provide binder length (known from your pipeline = 256)
    binder_length = 256
    ref_split = split_fused_chain(ref_model, "A", binder_length)
    """
    io = PDBIO()
    io.set_structure(ref_split)
    io.save(f"out_test_{bb_name}.pdb")
    """
    ref_target = ref_split[0]["A"]
    ref_binder = ref_split[0]["C"]
    ref_ligand = ref_split[0]["F"]
    
    mov_target = mov_model["A"]
    mov_binder = mov_model["B"]
    mov_ligand = mov_model["FUN"]
    
    

    # -------------------------------------------------------------
    # 3) Helper: get ordered CA coordinate lists based on sequence
    # -------------------------------------------------------------
    def get_ca_atoms(chain):
        atoms = []
        for res in chain:
            if is_aa(res, standard=True) and "CA" in res:
                atoms.append(res["CA"])
        return atoms
    
    def get_ca_coords(chain):
        coords = []
        for res in chain:
            if is_aa(res, standard=True) and "CA" in res:
                coords.append(res["CA"].get_coord())
        return np.array(coords)
    # compute full rmsd with full alignment :
    
    ref_ca = get_ca_atoms(ref_target) + get_ca_atoms(ref_binder)
    mov_ca = get_ca_atoms(mov_target) +get_ca_atoms(mov_binder)
    
    # ensure same length by trimming the longer one (safer than trusting numbering)
    L = min(len(ref_ca), len(mov_ca))
    ref_ca = ref_ca[:L]
    mov_ca = mov_ca[:L]
    
    sup = Superimposer()
    sup.set_atoms(ref_ca, mov_ca)
    rot, tran = sup.rotran
    full_rmsd = sup.rms
    data["full_rmsd"]=round(full_rmsd,2)



    # -------------------------------------------------------------
    # 4) Align targets based on CA atoms
    # -------------------------------------------------------------
    ref_ca = get_ca_atoms(ref_target)
    mov_ca = get_ca_atoms(mov_target)
    
    # ensure same length by trimming the longer one (safer than trusting numbering)
    L = min(len(ref_ca), len(mov_ca))
    ref_ca = ref_ca[:L]
    mov_ca = mov_ca[:L]
    
    sup = Superimposer()
    sup.set_atoms(ref_ca, mov_ca)
    rot, tran = sup.rotran
    target_rmsd = sup.rms
    data["target_rmsd"]=round(target_rmsd,2)

    
    # -------------------------------------------------------------
    # 5) Apply transform to ALL atoms in the moving structure
    # -------------------------------------------------------------
    for atom in mov_struct.get_atoms():
        atom.transform(rot, tran)
    
    mov_model = mov_struct[0]
    ref_target = ref_split[0]["A"]
    ref_binder = ref_split[0]["C"]
    ref_ligand = ref_split[0]["F"]
    
    mov_target = mov_model["A"]
    mov_binder = mov_model["B"]
    mov_ligand = mov_model["FUN"]
    # -------------------------------------------------------------
    # 6) Compute unaligned binder and ligand RMSDs
    # -------------------------------------------------------------

    #  version: from bindcraft
    def ca_map(chain):
        out = {}
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            if "CA" in res:
                hetflag, resseq, icode = res.get_id()
                out[(resseq, icode)] = res["CA"]
        return out
    
    ref_ca = ca_map(ref_binder)
    mov_ca = ca_map(mov_binder)
    
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
    binder_rmsd = float(np.sqrt((diffs * diffs).sum(axis=1).mean()))
    print(round(binder_rmsd, 2))
    data["binder_rmsd"]=round(binder_rmsd, 2)

    
    
    # ligand atoms: compute RMSD over all heavy atoms
    ref_lig_atoms = np.array([a.get_coord() for a in ref_ligand.get_atoms()])
    mov_lig_atoms = np.array([a.get_coord() for a in mov_ligand.get_atoms()])
    L = min(len(ref_lig_atoms), len(mov_lig_atoms))
    diffs=ref_lig_atoms[:L]-mov_lig_atoms[:L]
    lig_rmsd = float(np.sqrt((diffs * diffs).sum(axis=1).mean()))
    data["lig_rmsd"]=round(lig_rmsd, 2)
    
    
    # -------------------------------------------------------------
    # 7) Output
    # -------------------------------------------------------------
    print("Aligned target RMSD :", round(target_rmsd, 3))
    print("Binder RMSD (no align):", round(binder_rmsd, 3))
    print("Ligand RMSD (no align):", round(lig_rmsd, 3))
    # add metrics to csv file
    df = pd.DataFrame([data])
    df.to_csv(f"{AF3_struct}/design_confidences.csv", mode="a", index=False, header=not pd.io.common.file_exists(f"{AF3_struct}/design_confidences.csv"))
    
    
