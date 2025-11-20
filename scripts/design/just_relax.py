from colabdesign.af.alphafold.common import protein
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import residue_constants
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

from colabdesign.af.alphafold.common import protein
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import residue_constants

import pdbfixer
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--pdb", required=True, type=str, help="Input PDB")

INPUT_PDB = args.pdb



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



# running the relaxation +...
pdb_name = os.path.basename(INPUT_PDB).replace(".pdb", "")
relaxed_pdb_str=relax_me(INPUT_PDB, f"{pdb_name}relaxed.pdb")
