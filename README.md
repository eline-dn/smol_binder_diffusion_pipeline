# De novo *Chemically Induced Dimer* design pipeline using RFdiffusionAA, pMPNN & co

*Designing binders for protein+ ligand complexes*

Custom modifications of this pipeline https://github.com/ikalvet/heme_binder_diffusion from Indrek Kalvet, PhD (Institute for Protein Design, University of Washington).

Their pipeline consists of 7 steps:
0) The protein backbones are generated with RFdiffusionAA
1) Sequence is designed with proteinMPNN (without the ligand)
2) Structures are predicted with AlphaFold2
3) Ligand binding site is designed with LigandMPNN/FastRelax, or Rosetta FastDesign
4) Sequences surrounding the ligand pocket are diversified with LigandMPNN
5) Final designed sequences are predicted with AlphaFold2
6) Alphafold2-predicted models are relaxed with the ligand and analyzed

Ours will be a bit different, in order to be able to design Chemically induced dimers: Design a binder that binds to a target protein + small molecule complex. The designed binder need to be highly specific for the prot + ligand complex, and shouldn't bind to the target protein only or the ligand only. Target protein can stabilize the binder + ligand interaction. 

0) Binder backbone generation with RFdiffusion all_atoms, scaffolding the target + ligand complex (i.e. generating the backbone on top of the target, on its C-term or N-term).
1) Generating a sequence for the binder with protein MPNN. The whole output from the previous step (target + ligand + binder backbone) is used, but only the binder sequence is redesigned.
2) The generated binders are repredicted with AF2 from their sequence, and their structure is re-aligned to their initialy diffused backbone, in order to see if a sequence exists for the diffused backbone, and if this binder can fold. Binders are filtered based on their plDDT and their RMSD to the initial predicted backbone.
3) Ligand-pocket redesign with ligand MPNN. 



The begining is quite similar to the initial pipeline.
steps 0 to 2 are done with the python script run.py, slightly adapted from the ipynb file (do not run as it is, proceed step by step
the main differences with the original pipeline is the ability to run it in my local env, and for the purpose of designing a binder to a target + small molecule (e.g. trimming of some of the diffusion and pMPNN reference files to filter the binders)








## Installation
### Dependencies

#### LigandMPNN and AlphaFold2
To download the LigandMPNN and AlphaFold2 (v2.3.2) repositories referenced in this pipeline run:
```
git submodule init
git submodule update
```

To download the model weight files for AlphaFold2 and proteinMPNN run this command:<br>
`bash get_af2_and_mpnn_model_params.sh`

If you already have downloaded the weights elsewhere on your system then please edit these scripts with appropriate paths:<br>
    proteinMPNN: `lib/LigandMPNN/mpnn_api.py` [lines 45-49]<br>
    AlphaFold2: `scripts/af2/AlphaFold2.py` [line 40]

#### RFdiffusionAA:
Download RFdiffusionAA from here: https://github.com/baker-laboratory/rf_diffusion_all_atom<br>
and follow its instructions.<br>
Make sure to provide a full path to the checkpoint file in this configuration file:<br>
`rf_diffusion_all_atom/config/inference/aa.yaml`

#### RFjoint inpainting (proteininpainting)
(Optional) Download RFjoint Inpainting here: https://github.com/RosettaCommons/RFDesign

Inpainting is used to further resample/diversify diffusion outputs, and it may also increase AF2 success rates.

### Python or Apptainer image
This pipeline consists of multiple different Python scripts using a different Python modules - most notably PyTorch, PyRosetta, Jax, Jaxlib, Tensorflow, Prody, OpenBabel.<br>
Separate conda environments for AlphaFold2 and RFdiffusionAA/ligandMPNN were used to test this pipeline, and the environment YML files are provided in `envs/`.


To create a conda environment capable of running RFdiffusionAA, LigandMPNN and PyRosetta, set it up as follows:<br>
`conda env create -f envs/diffusion.yml`


A minimal conda environment for AlphaFold2 is set up as follows:<br>
`conda env create -f envs/mlfold.yml`


### Executing the pipeline
Please adjust the the critical paths defined in the first couple of cells of the notebook based on your system configuration. Other than that, the pipeline is executed by running the cells and waiting for them to finish.

Certain tasks are configured to run as Slurm jobs on a compute cluster. The Slurm script setup is handled in `scripts/utils/utils.py` by the function `create_slurm_submit_script()`.
Please modify this script, and any references to it in the notebook, based on how your system accepts jobs.
