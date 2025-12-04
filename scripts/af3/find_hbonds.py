from Bio.PDB import PDBIO
import os
import pyrosetta as pyr
import pyrosetta.rosetta
import numpy as np
from pyrosetta.rosetta.core.scoring import fa_rep
import os, sys
import pandas as pd
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


SCRIPT_PATH = os.path.dirname(__file__)
sys.path.append(f"{SCRIPT_PATH}/../../utils")

import Bio.PDB
#PDB_parser = Bio.PDB.PDBParser(QUIET=True)
CIF_parser = Bio.PDB.MMCIFParser(QUIET=True)

def get_angle(a1, a2, a3):
    a1 = np.array(a1)
    a2 = np.array(a2)
    a3 = np.array(a3)

    ba = a1 - a2
    bc = a3 - a2

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return round(np.degrees(angle), 1)



def find_hbonds_to_residue_atom(pose, lig_seqpos, target_atom): # the one actually used in the scoring script
    """
    Counts how many Hbond contacts input atom has with the protein.
    """
    HBond_res = 0

    for res in pose.residues:
        if res.seqpos() == lig_seqpos or res.is_ligand():
            break
        if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
            for polar_H in res.Hpos_polar():
                if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz(polar_H)).norm() < 2.5:
                    # If the polar atom is from the backbone then check that the X-H...Y angle is close to linear.
                    # It is assumed that polar backbone H is only attached to backbone N
                    if res.atom_is_backbone(polar_H):
                        print("infos hscores for target atom:",target_atom,res.seqpos(), target_atom, res.atom_name(polar_H), get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)))
                        if get_angle(res.xyz("N"), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)) < 140.0:
                            continue
                    HBond_res += 1
                    break
    return HBond_res

"""
def find_hbonds_from_prot(pose, lig_seqpos, target_atom): # test
    #Counts how many Hbond contacts input atom has with the protein.
    HBond_res = 0

    for res in pose.residues:
        if res.seqpos() == lig_seqpos or res.is_ligand():
            break
        if # res in chain A: 
            continue
        if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz('CA')).norm() < 10.0:
            #for atoms in res. # find the residues's  atoms
                if atoms in # list of atoms capable of receiving an H in a hydrogen bond
                if (pose.residue(lig_seqpos).xyz(target_atom) - res.xyz(polar_H)).norm() < 2.5:
                    # If the polar atom is from the backbone then check that the X-H...Y angle is close to linear.
                    # It is assumed that polar backbone H is only attached to backbone N
                    if res.atom_is_backbone(polar_H):
                        print(res.seqpos(), target_atom, res.atom_name(polar_H), get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)))
                        if get_angle(res.xyz("N"), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)) < 140.0:
                            continue
                    HBond_res += 1
                    break
    return HBond_res

"""
# another suggestion: to be tested
from pyrosetta import *
from pyrosetta.rosetta.core.scoring.hbonds import HBondSet
from pyrosetta.rosetta.core.id import AtomID

def find_hbonds_ligand_atom_to_chainB(pose, lig_idx, atom_name="A1", target_chain="B"):
    """
    Returns all H-bonds involving the ligand atom 'atom_name'
    where the partner atom belongs to residues in chain B.
    """
    # --- 1. Build the HBondSet for the pose ---------------------
    hbset = HBondSet()
    hbset.setup_for_residue_pair_energies(pose, False)

    # --- 2. Locate ligand AtomID --------------------------------
    lig_res = pose.residue(lig_idx)
    if not lig_res.has(atom_name):
        raise ValueError(f"Ligand residue {lig_idx} has no atom named '{atom_name}'")

    atom_id = AtomID(lig_res.atom_index(atom_name), lig_idx)

    # --- 3. Find all H-bonds involving this atom ----------------
    hbonds = hbset.atom_hbonds_all(atom_id)

    results = []

    for hb in hbonds:
        don_res = hb.don_res()   # donor residue index
        acc_res = hb.acc_res()   # acceptor residue index

        # Identify which side is ligand and which is protein
        if don_res == lig_idx:
            partner = acc_res
            donor_is_ligand = True
        elif acc_res == lig_idx:
            partner = don_res
            donor_is_ligand = False
        else:
            continue  # should not happen, atom_hbonds_all already filtered to relevant atom

        # Check partner is in chain B
        if pose.pdb_info().chain(partner) != target_chain:
            continue

        # Collect useful info
        partner_res = pose.residue(partner)
        partner_atom = (hb.acc_atm() if donor_is_ligand else hb.don_atm())
        partner_atom_name = partner_res.atom_name(partner_atom).strip()

        results.append({
            "lig_atom": atom_name,
            "lig_idx": lig_idx,
            "partner_resi": partner,
            "partner_chain": target_chain,
            "partner_atom": partner_atom_name,
            "lig_is_donor": donor_is_ligand,
            "energy": hb.energy(),
            "distance": hb.distance()
        })

    return results


# ------------------ Example usage ------------------

# pose = pyrosetta.pose_from_file("complex_with_ligand.pdb")
# lig_idx = 250  # example ligand position

# hb_info = find_hbonds_ligand_atom_to_chainB(pose, lig_idx, "A1", "B")
# for h in hb_info:
#     print(h)



#-------------------------------

# Ligand information
params = [f"/work/lpdi/users/eline/smol_binder_diffusion_pipeline/input/FUN.params"]  # Rosetta params file(s)
LIGAND = "FUN"


parser = argparse.ArgumentParser()
parser.add_argument("--pdb", nargs="+", type=str, help="Input PDB") # list of pdb files 
#parser.add_argument("--params", nargs="+", type=str, help="Params files")
args = parser.parse_args()


"""
Getting PyRosetta started
"""
extra_res_fa = ""
if True: #args.params is not None:
    extra_res_fa = "-extra_res_fa"
    for p in params:
        extra_res_fa += f" {p}"

NPROC = os.cpu_count()
if "SLURM_CPUS_ON_NODE" in os.environ:
    NPROC = os.environ["SLURM_CPUS_ON_NODE"]
elif "OMP_NUM_THREADS" in os.environ:
    NPROC = os.environ["OMP_NUM_THREADS"]


DAB = f"{SCRIPT_PATH}/../utils/DAlphaBall.gcc" # This binary was compiled on UW systems. It may or may not work correctly on yours
assert os.path.exists(DAB), "Please compile DAlphaBall.gcc and manually provide a path to it in this script under the variable `DAB`\n"\
                        "For more info on DAlphaBall, visit: https://www.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Filters/HolesFilter"


pyr.init(f"{extra_res_fa} -dalphaball {DAB} -beta_nov16 -run:preserve_header -mute all "
         f"-multithreading true -multithreading:total_threads {NPROC} -multithreading:interaction_graph_threads {NPROC}")
df_scores = pd.DataFrame()

for i,INPUT_PDB in enumerate(args.pdb): # actually some mmcif files that we convert to pdb for pyrosetta
    structure = CIF_parser.get_structure("x", INPUT_PDB)
    old_id = "FUN"
    new_id = "F"
    chain=structure[0][old_id]
    chain.id = new_id
    for res in structure.get_residues():
        if res.resname.strip().startswith("LIG"):  # or your rule
            res.resname = "FUN"
    io = PDBIO()
    io.set_structure(structure)
    pdbfile=INPUT_PDB.replace(".cif", ".pdb") # convert mmcif to pdb
    io.save(pdbfile)
    
    input_pose = pyrosetta.pose_from_file(pdbfile)
    pose = input_pose.clone()
    ligand_resno = pose.size()
    print("lig pos:",ligand_resno)
    assert pose.residue(ligand_resno).is_ligand()
    
    # Using a custom function to find HBond partners of the groups that might be involved
    # build ligand pose 
    ligand_pose = pyrosetta.rosetta.core.pose.Pose()
    pyrosetta.rosetta.core.pose.append_subpose_to_pose(ligand_pose, pose, pose.size(), pose.size(), 1)
    # we will look for these atoms: 
    at_list=list(("O1", "O2","N1","O3", "O5", "O4"))
    for n in at_list:
        df_scores.at[i, f"{n}_hbond"] = find_hbonds_to_residue_atom(pose, ligand_resno, n) # this function Counts how many Hbond contacts input atom has with the protein.
        # the target atoms have to be adapted to the ligand

    if any([df_scores.at[i, x] > 0.0 for x in ['N1_hbond','O1_hbond','O2_hbond', 'O3_hbond','O5_hbond','O4_hbond']]):
        df_scores.at[i, 'binder_hbond'] = True
    else:
        df_scores.at[i, 'binder_hbond'] = False
    print("h_scores:",df_scores)
    # test the other hbond function
    h_list=list(("H1","H9","H10", "H11")) 
    for h in h_list:
        print("find hbonds with h:",h)
        hb_info = find_hbonds_ligand_atom_to_chainB(pose, ligand_resno, h, "B")
    for h in hb_info:
         print("hb infos:",h)

    """
    test
    res.OOC()
    res.OH() """
    ## Calculating shape complementarity between binder and target
    #lig_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(ligand_seqpos)
    target_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("A")
    binder_sel = pyrosetta.rosetta.core.select.residue_selector.ChainSelector("B")
    sc = pyrosetta.rosetta.protocols.simple_filters.ShapeComplementarityFilter()
    sc.use_rosetta_radii(True)
    sc.selector1(target_sel)
    sc.selector2(binder_sel)
    df_scores.at[i, "sc"] = sc.score(pose)
    df_scores.at[i,"cif_path"]=INPUT_PDB
    #see also: bindcraft's score_interface function in pyrosetta_utils.py
    #interfacescore = iam.get_all_data()
    #interface_sc = interfacescore.sc_value # shape complementarity
    #interface_interface_hbonds = interfacescore.interface_hbonds # number of interface H-bonds

df_scores.to_csv("h_scores.csv")

