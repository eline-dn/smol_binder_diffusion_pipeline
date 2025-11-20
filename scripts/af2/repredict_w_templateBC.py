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


def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, trajectory_pdb, prediction_models,  design_paths, seed=None):
    prediction_stats = {}

    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())

    # reset filtering conditionals
    pass_af2_filters = True
    filter_failures = {}


    # start prediction per AF2 model, 2 are used by default due to masked templates
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
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

           
      

    # AF2 filters passed, contuing with relaxation
    for model_num in prediction_models:


    return prediction_stats, pass_af2_filters




clear_mem()
params=
complex_pdb= # needs to be split in order to separate binder from target (change chain label for binder residues in pMPNN output)
binder_length=

# compile complex prediction model
complex_prediction_model = mk_afdesign_model(protocol="binder", num_recycles=3, data_dir=params, 
                                            use_multimer=True,
                                             use_initial_guess=False, #Introduce bias by providing binder atom positions as a starting point for prediction.
                                             use_initial_atom_pos=False # Introduce atom position bias into the structure module for atom initilisation.


#if advanced_settings["predict_initial_guess"] or advanced_settings["predict_bigbang"]:

complex_prediction_model.prep_inputs(pdb_filename=complex_pdb,
                                         chain='A',
                                         binder_chain='B',
                                         binder_len=binder_length,
                                         use_binder_template=True,
                                         rm_target_seq=False, #remove target template sequence for reprediction (increases target flexibility)
                                        rm_template_ic=True)
"""
else:
    complex_prediction_model.prep_inputs(pdb_filename=target_settings["starting_pdb"],
                                         chain=target_settings["chains"], 
                                         binder_len=length,
                                         rm_target_seq=advanced_settings["rm_template_seq_predict"])
"""

  
design_name=

mpnn_complex_statistics, pass_af2_filters = predict_binder_complex(complex_prediction_model,
                                                                    mpnn_sequence['seq'], design_name,
                                                                
                                                                     trajectory_pdb, prediction_models,
                                                                     design_paths)
