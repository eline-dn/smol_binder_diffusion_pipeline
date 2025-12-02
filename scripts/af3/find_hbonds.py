"""
Extra modules for scoring protein structures
Authors: Chris Norn, Indrek Kalvet
"""
import os
import pyrosetta
import pyrosetta.rosetta
import numpy as np
from pyrosetta.rosetta.core.scoring import fa_rep

import Bio.PDB
BIO_PDB_parser = Bio.PDB.PDBParser(QUIET=True)


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
                        # print(res.seqpos(), target_atom, res.atom_name(polar_H), get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)))
                        if get_angle(res.xyz(1), res.xyz(polar_H), pose.residue(lig_seqpos).xyz(target_atom)) < 140.0:
                            continue
                    HBond_res += 1
                    break
    return HBond_res


# Using a custom function to find HBond partners of the groups that might be involved
    # we will look for these atoms: 
    at_list=list(("N2","O3"))
    for n in at_list:
        df_scores.at[0, f"{n}_hbond"] = scoring_utils.find_hbonds_to_residue_atom(pose, ligand_seqpos, n) # this function Counts how many Hbond contacts input atom has with the protein.
        # the target atoms have to be adapted to the ligand


    # Checking h_bonds: what atoms need to have hbonds with the structure: N2 and O3 are the ones sticking out of the initial pocket
    # maybe just counting them? and setting a threshold 
    
    if any([df_scores.at[0, x] > 0.0 for x in ['N2_hbond', 'O3_hbond']]):
        df_scores.at[0, 'binder_hbond'] = 1.0
    else:
        df_scores.at[0, 'binder_hbond'] = 0.0
    
