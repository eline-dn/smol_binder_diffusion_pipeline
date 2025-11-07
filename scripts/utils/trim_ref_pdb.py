import os, sys, glob
import pandas as pd
import numpy as np

from Bio import BiopythonWarning
from Bio.PDB import DSSP, Selection, Polypeptide, Select, Chain, Superimposer
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities

from Bio.PDB import PDBParser, PDBIO, Model, Chain, Structure
from Bio.PDB import StructureBuilder
from Bio.PDB.Polypeptide import is_aa # Assuming is_aa is needed and available


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
    

def trim_pdb(input_pdb_path, output_pdb_path, trim_length=412):
    """
    Trims the first N amino acids from a PDB file.

    Args:
        input_pdb_path (str): Path to the input PDB file.
        output_pdb_path (str): Path to save the trimmed PDB file.
        trim_length (int): The number of amino acids to trim from the beginning.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb_path)

    for model in structure:
        for chain in model:
            # Get all residues in the chain
            residues = list(chain.get_residues())
            # Keep residues from index trim_length onwards
            for i, residue in enumerate(residues):
                if i < trim_length:
                    chain.detach_child(residue.get_id())

    io = PDBIO(use_model_flag=1)
    io.set_structure(structure)
    io.save(output_pdb_path)

# Example usage:
# Assuming you have a PDB file named 'input.pdb' in the current directory
# trim_pdb('input.pdb', 'trimmed_output.pdb')
# print("Trimmed PDB file saved as 'trimmed_output.pdb'")



def extract_chain(input_pdb_path: str, output_pdb_path: str, chain_id: str):
    """
    Extracts a specific chain from a PDB file using _copy_structure_with_only_chain
    and saves it to a new PDB file with explicit MODEL/ENDMDL records.

    Args:
        input_pdb_path (str): Path to the input PDB file (complex).
        output_pdb_path (str): Path to save the extracted chain PDB file.
        chain_id (str): The identifier of the chain to extract (e.g., "A", "B").
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb_path)
    io = PDBIO(use_model_flag=1)

    # Use the helper function to get a new structure with only the desired chain
    new_structure = _copy_structure_with_only_chain(structure, chain_id)

    # --- Debug Print Statements ---
    print(f"--- Debug: Saving structure for {output_pdb_path} ---")
    print(f"Number of models in structure to save: {len(new_structure)}")
    for i, model in enumerate(new_structure):
        print(f"  Model {i}: Number of chains = {len(model)}")
        for j, chain in enumerate(model):
            print(f"    Chain {chain.id}: Number of residues = {len(chain)}")
            # Optional: print a few residue IDs and segids to confirm content
            print(f"      First few residues (ID, SegID): {[(r.id, r.segid) for r in list(chain.get_residues())[:5]]}")
    print("----------------------------------------------------")
    # --- End Debug Print Statements ---

    # Save the new structure, explicitly writing model records
    io.set_structure(new_structure)
    io.save(output_pdb_path)

# Example usage (assuming you have a complex PDB file named 'complex.pdb'):
# extract_chain('complex.pdb', 'chain_A.pdb', 'A')
# print("Chain A saved to 'chain_A.pdb'")

pdb_files=sys.argv[1] # path to the folder with the input/ reference pdb files, that we want to extract the binder structure from
output_folder=sys.argv[2] # path to the folder where the trimmed pdb should be store (= binder structure references)

for pdb in glob.glob(f"{pdb_files}/*.pdb"):
  out_path=os.path.join(output_folder, os.path.basename(pdb))
  extract_chain(pdb, out_path, 'A')
  trim_pdb(out_path, out_path)


print("Done")

