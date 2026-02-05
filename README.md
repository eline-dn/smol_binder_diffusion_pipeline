# De novo *Chemically Induced Dimers* design pipeline using RFdiffusionAA, pMPNN & co

*Designing binders for protein+ligand complexes*

Custom modifications of this pipeline https://github.com/ikalvet/heme_binder_diffusion from Indrek Kalvet, PhD (Institute for Protein Design, University of Washington).

Their pipeline consists of 7 steps:
0) The protein backbones are generated with RFdiffusionAA
1) Sequence is designed with proteinMPNN (without the ligand)
2) Structures are predicted with AlphaFold2
3) Ligand binding site is designed with LigandMPNN/FastRelax, or Rosetta FastDesign
4) Sequences surrounding the ligand pocket are diversified with LigandMPNN
5) Final designed sequences are predicted with AlphaFold2
6) Alphafold2-predicted models are relaxed with the ligand and analyzed

Ours will be a bit different, in order to be able to design Chemically induced dimers: Design a binder that binds to a target protein + small molecule complex. The designed binder need to be highly specific for the prot + ligand complex, and shouldn't bind to the target protein only or the ligand only. Target protein can stabilize the binder + ligand interaction. We will focus here on two targets, known as 3DGQ and 1Z9Y in the PDB. 

0) Binder backbone generation with RFdiffusion all_atoms, scaffolding the target + ligand complex (i.e. generating the backbone on top of the target, on its C-term or N-term).
1) Generating a sequence for the binder with protein MPNN. The whole output from the previous step (target + ligand + binder backbone) is used, but only the binder sequence is redesigned.
2) Checking backbone designability : The generated binders are repredicted with AF2 from their sequence, and their structure is re-aligned to their initialy diffused backbone, in order to see if a sequence exists for the diffused backbone, and if this binder can fold. Binders are filtered based on their plDDT and their RMSD to the initial predicted backbone.
3) Ligand-pocket redesign /partial redesign /or whole binder redesign with ligand MPNN.
4) scoring pipeline (see CID_scoring github repo), with binary and ternary cmplexes reprediction and analysis. 



The begining is quite similar to the initial pipeline.

the main differences with the original pipeline is the ability to run it in my local env, and for the purpose of designing a binder to a target + small molecule (e.g. trimming of some of the diffusion and pMPNN reference files to filter the binders)
Running these steps smoothly is currently WIP with 'run_phase1.py', ....


## Running the pipeline 

1) Backbone design with RF diffusion all atoms:
Running with different combinations of inference.deterministic, inference.ppi_design and inference.scaffold_guided combinations

```sbatch 0_rf_diff.sh```

2) Sequence design with protein MPNN
```python 1_pMPNN.py```
Wait for the launched slurm script to finish. Sequence fasta files are in the './1_protein_mpnn/seqs' folder, backbones with threaded sequence in './1_protein_mpnn/backbones'

3) Filtering for backbone designability with AF2

The sequences from binders only are extracted from the fasta files and repredicted with AF2 in single sequence mode:

Run ```2.1_prep_AF2_inputs.py``` to 
Add 
```
module load gcc/13.2.0
module load cuda/12.4.1
```

at the begining of the produced script and ```sbatch submit_af2.sh```

The outputs are then filtered to check if the sequence folds into the intended backbone: 
TO DO verify and mb re-run this step (2.2_filter_af2_out.py) ? check how references are mapped to repredictions in /scripts/utils/analyze_af2.py and if the param argument is necessary

good binders in {AF2_DIR}/good/ 

4) Pocket and sequence redesign with ligand MPNN

=> with simple_redesign.py in run_alt_phase2.py
To DO: re-run this step with appropriate names and save pdb files by threading the new sequence on the pMPNN backbones, splitting them into chain A and B and renaming the ligand chain to L 
=> then use as binder refs for the scoring pipeline 

5) Binder scoring pipeline

See https://github.com/eline-dn/CID_scoring 




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


### Python or Apptainer image
This pipeline consists of multiple different Python scripts using a different Python modules - most notably PyTorch, PyRosetta, Jax, Jaxlib, Tensorflow, Prody, OpenBabel.<br>
Separate conda environments for AlphaFold2 and RFdiffusionAA/ligandMPNN were used to test this pipeline, and the environment YML files are provided in `envs/`.


To create a conda environment capable of running RFdiffusionAA, LigandMPNN and PyRosetta, set it up as follows:<br>
`conda env create -f envs/diffusion.yml`


A minimal conda environment for AlphaFold2 is set up as follows:<br>
`conda env create -f envs/mlfold.yml`



