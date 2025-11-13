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


# SETUP-----------------------------------------------------

diffusion_script = "/work/lpdi/users/eline/rf_diffusion_all_atom/run_inference.py"  # edit this
# inpaint_script = "PATH/TO/RFDesign/inpainting/inpaint.py"  # edit this if needed
proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"  # from submodule
AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"  # from submodule


### Python and/or Apptainer executables needed for running the jobs
### Please provide paths to executables that are able to run the different tasks.
### They can all be the same if you have an environment with all of the necessary Python modules in one

CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    # "af2":"/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
}

username = getpass.getuser()  # your username on the running system
EMAIL = "eline.denis@epfl.ch"  # edit based on your organization. For Slurm job notifications.

PROJECT = "CID_1Z9Y"

### Path where the jobs will be run and outputs dumped
WDIR = "/work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout"

if not os.path.exists(WDIR):
    os.makedirs(WDIR, exist_ok=True)

print(f"Working directory: {WDIR}")

# Ligand information
params = [f"{SCRIPT_DIR}/theozyme/HBA/HBA.params"]  # Rosetta params file(s) !!!!!!!!!!!!!!!!!!!! to do: check if this is used in analyze af2 and remove if not. if yes, modify it according to the ligand!
LIGAND = "FUN"


DIFFUSION_DIR = f"{WDIR}/0_diffusion"
MPNN_DIR = f"{WDIR}/1_proteinmpnn"
AF2_DIR = f"{WDIR}/2_af2"


# to add: parse script parameters with add parser to know wich steps need to be done


#--------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------0: diffusion run------------------------------------------------------------------------------------------------------

## setting up diffusion run: doesn't run diffusion, just to set up the directories
    # diffusion_inputs = glob.glob(f"{SCRIPT_DIR}/input/*.pdb")
diffusion_inputs = list()
diffusion_inputs.append("1Z9Y_clean.pdb")
print(f"Found {len(diffusion_inputs)} PDB files")

os.chdir(DIFFUSION_DIR)



## diffusion run if not done
if do["0_diffusion"]:
    """ in the shell:sbatch nd[1-5]run_inference.slurm
    write different commands, testing different parameters
    
    ## Analyzing locally
    ### to do: find a way to parallelize this!!
    p = subprocess.Popen(analysis_command, shell=True)
    (output, err) = p.communicate()
    
    """




# ----------------1: Running ProteinMPNN on diffused backbones ---------------------------------------------------
#pdbs should be in /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/0_diffusion, ie in DIFFUSION_DIR
#pattern = re.compile(r"t2_\d+_(2|[2-8]\d|89)\.pdb$")
pattern = re.compile(r"t2_\d+_(1|[3-9]|1[0-9]|9[0-9]|1[0-4][0-9])\.pdb$")
all_pdbs = glob.glob(f"{DIFFUSION_DIR}/t2_*.pdb")
diffused_backbones_good = [f for f in all_pdbs if pattern.search(f)]

assert len(diffused_backbones_good) > 0, "No good backbones found!"

os.chdir(WDIR)

MPNN_DIR = f"{WDIR}/1_proteinmpnn"
os.makedirs(MPNN_DIR, exist_ok=True)
os.chdir(MPNN_DIR)



if  do["1proteinmpnn"]:
  """the creation of the mask dict from the trb file allow us to use pMPNN on the backbone pdb file from rf diff, only the binder will be redesigned.
  Parsing diffusion output TRB files to extract fixed motif residues.
  These residues will not be redesigned with proteinMPNN
  """
  mask_json_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/design/make_maskdict_from_trb.py --out masked_pos.jsonl --trb"
  for d in diffused_backbones_good:
    mask_json_cmd += " " + d.replace(".pdb", ".trb")
  
  p = subprocess.Popen(mask_json_cmd, shell=True)
  (output, err) = p.communicate()
  assert os.path.exists("masked_pos.jsonl"), "Failed to create masked positions JSONL file"
  
  MPNN_temperatures = [0.1, 0.2, 0.3]
  MPNN_outputs_per_temperature = 5
  MPNN_omit_AAs = "CM"
  
  commands_mpnn = []
  cmds_filename_mpnn = "commands_mpnn"
  with open(cmds_filename_mpnn, "w") as file:
    for T in MPNN_temperatures:
        for f in diffused_backbones_good:
            commands_mpnn.append( ### !!!! here don't forget to change the output folder if needed!
                f"{PYTHON['proteinMPNN']} {proteinMPNN_script} "
                f"--model_type protein_mpnn --ligand_mpnn_use_atom_context 0 --file_ending _T{T} "
                "--fixed_residues_multi masked_pos.jsonl --out_folder ./part2 " 
                f"--number_of_batches {MPNN_outputs_per_temperature} --temperature {T} "
                f"--omit_AA {MPNN_omit_AAs} --pdb_path {f} "
                f"--checkpoint_protein_mpnn {SCRIPT_DIR}/lib/LigandMPNN/model_params/proteinmpnn_v_48_020.pt\n"
            )
            file.write(commands_mpnn[-1])
  print("Number of proteinMPNN commands:", len(commands_mpnn))
  print("Example MPNN command:")
  print(commands_mpnn[-1])

"""  TO DO: run this locally (the python script is in a slurm job itself but in a parallelized way
  submit_script = "submit_mpnn.sh"
utils.create_slurm_submit_script(
    filename=submit_script,
    name="1_proteinmpnn",
    mem="4g",
    N_cores=1,
    time="0:45:00",
    partition="h100",
    email=EMAIL,
    array=len(commands_mpnn),
    array_commandfile=cmds_filename_mpnn,
    group=150,
)

p = subprocess.Popen(["sbatch", submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(output, err) = p.communicate()
""" 

#--------------------------------------------------2: Running AlphaFold2-------------------------------------------------

AF2_DIR = f"{WDIR}/2_af2"
os.makedirs(AF2_DIR, exist_ok=True)
os.chdir(AF2_DIR)

#2.1 extract binder sequences (trimm the fasta files from pMPNN)

if do["2.1af2_trim_fastas"]:
  fasta_files = glob.glob(f"{MPNN_DIR}/part2/seqs/*_T0.*.fa") ### output file
  output_dir = f"{AF2_DIR}/trimmed_fastas_2"
  os.makedirs(output_dir, exist_ok=True)

  for ff in fasta_files:
      with open(ff, "r") as f:
          lines = f.readlines()
      header = ""
      sequence = ""
      output_filename = os.path.join(output_dir, os.path.basename(ff))
      with open(output_filename, "w") as outfile:
          for line in lines:
              if line.startswith(">"):
                  if sequence:
                      trimmed_sequence = sequence[:-256]
                      outfile.write(header + trimmed_sequence + "\n")
                  header = line.strip() + "\n"
                  sequence = ""
              else:
                  sequence += line.strip()
          if sequence:
              trimmed_sequence = sequence[:-256]
              outfile.write(header + trimmed_sequence + "\n")
      print(f"Processed and trimmed: {ff} -> {output_filename}")
  done["2.1af2_trim_fastas"] = True

# --------------------------2.2 Running AF2 binder strucutre repredictions -------------------------------------------
if (do["2af2_binder_prediction"]) and done["2.1af2_trim_fastas"]:
  ### First collecting MPNN outputs and creating FASTA files for AF2 input
  mpnn_fasta = utils.parse_fasta_files(glob.glob(f"{AF2_DIR}/trimmed_fastas_2/*.fa"))
  mpnn_fasta = {k: seq.strip() for k, seq in mpnn_fasta.items() if "model_path" not in k}  # excluding the diffused poly-A sequence
  # Giving sequences unique names based on input PDB name, temperature, and sequence identifier
  mpnn_fasta = {k.split(",")[0]+"_"+k.split(",")[2].replace(" T=", "T")+"_0_"+k.split(",")[1].replace(" id=", ""): seq for k, seq in mpnn_fasta.items()}
  print(f"A total of {len(mpnn_fasta)} sequences will be predicted.")
  ## Splitting the MPNN sequences based on length
  ## and grouping them in smaller batches for each AF2 job
  ## Use group size of >40 when running on GPU. Also depends on how many sequences and resources you have.
  SEQUENCES_PER_AF2_JOB = 100  # GPU
  mpnn_fasta_split = utils.split_fasta_based_on_length(mpnn_fasta, SEQUENCES_PER_AF2_JOB, write_files=True)
  ## Setting up AlphaFold2 run
  AF2_recycles = 3
  AF2_models = "4"  # add other models to this string if needed, i.e. "3 4 5"
  commands_af2 = []
  cmds_filename_af2 = "commands_af2"
  with open(cmds_filename_af2, "w") as file:
      for ff in glob.glob("*.fasta"):
          commands_af2.append(f"{PYTHON['af2']} {AF2_script} "
                            f"--af-nrecycles {AF2_recycles} --af-models {AF2_models} "
                            f"--fasta {ff} --scorefile {ff.replace('.fasta', '.csv')}\n")
          file.write(commands_af2[-1])
  
  print("Example AF2 command:")
  print(commands_af2[-1])
  print("Number of AF2 commands:")
  print(len(commands_af2))

  ### Running AF2 with Slurm.
  ### Running jobs on the GPU. It takes ~1-2 minutes per sequence
  ### !!! parallelization TO DO  

submit_script = "submit_af2.sh"
utils.create_slurm_submit_script(filename=submit_script, name="2_af2", mem="6g",
                                      N_cores=2, gpu=True, partition="h100", time="30:00:00", email=EMAIL, array=len(commands_af2),
                                      array_commandfile=cmds_filename_af2, group=25) ## don't forget to adjust group!
  # /!\ need to add:
  """
module load gcc/13.2.0
module load cuda/12.4.1
  """
  # to the submit script before the commands
  #if True: #not os.path.exists(AF2_DIR+"/.done"):
  #p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  #(output, err) = p.communicate()


#---------------------------------------------------2.3 Analyzing AF results ---------------------------------------------------

# Combining all CSV scorefiles into one
os.system("head -n 1 $(ls *aa*.csv | shuf -n 1) > scores.csv ; for f in *aa*.csv ; do tail -n +2 ${f} >> scores.csv ; done")
assert os.path.exists("scores.csv"), "Could not combine scorefiles"



if do["trim_pdb"]:
  # TO DO make the trimm pdb part more modulable, fuse the two scripts in one with a n_term or c_term arg
  ## need to trim the reference pdb files to compare binder wth binder, without the target + ligand:
  # remove up to aa 410 + ligand at the end (= extract chain A only)
  trim_cmd=f"{PYTHON['general']} {SCRIPT_DIR}/scripts/utils/trim_ref_pdb_nterm.py {DIFFUSION_DIR}/ {DIFFUSION_DIR}/filtered_structures/bindersonly "
  submit_script = "submit_ref_extraction.sh"
  utils.create_slurm_submit_script(filename=submit_script, name="binder_extraction",
                                      mem="16g", N_cores=8, partition="h100", time="0:05:00", email=EMAIL,
                                      command=trim_cmd, outfile_name="output_extraction")

  p = subprocess.Popen(["sbatch", submit_script])
  (output, err) = p.communicate()


### Calculating the RMSDs of filtered AF2 predictions relative to the diffusion outputs
### Catalytic residue sidechain RMSDs are calculated in the reference PDB has REMARK 666 line present

analysis_cmd = f"{PYTHON['general']} {SCRIPT_DIR}/scripts/utils/analyze_af2.py --scorefile scores.csv "\
               f"--ref_path {DIFFUSION_DIR}/bindersonly/ --mpnn --lddt 0.80 --params {' '.join(params)}"



    ## Running as a Slurm job TO DO: make is run locally, still paralellized
submit_script = "submit_af2_analysis.sh"
utils.create_slurm_submit_script(filename=submit_script, name="af2_analysis",
                                     mem="16g", N_cores=8, partition="h100", time="0:60:00", email=EMAIL,
                                     command=analysis_cmd, outfile_name="output_analysis")

p = subprocess.Popen(["sbatch", submit_script])
(output, err) = p.communicate()


###------------- Filtering AF2 scores based on rmsd and plDDT-------------
scores_af2 = pd.read_csv("scores.sc", sep="\s+", header=0)

scores_af2['lDDT'] = pd.to_numeric(scores_af2['lDDT'], errors='coerce')
scores_af2['rmsd']= pd.to_numeric(scores_af2['rmsd'], errors='coerce')

# plot distributions
## Plotting AF2 scores
plt.figure(figsize=(12, 3))
plt.hist(scores_af2['lDDT'])
plt.title('lDDT')
plt.xlabel('lDDT')
plt.savefig("score_plddt_hist.png")
plt.close()   

plt.figure(figsize=(12, 3))
plt.hist(scores_af2['rmsd'])
plt.title('rmsd')
plt.xlabel('rmsd')
plt.savefig("score_rmsd_hist.png")
plt.close()


plt.figure(figsize=(10, 5))
sns.scatterplot(data=scores_af2, x='lDDT', y='rmsd')
plt.xlabel('plDDT')
plt.ylabel('rmsd')
plddt = 80
plt.axvline(x=plddt, ymin=0, ymax=1, color="black", linestyle="--")

rmsd = 3
plt.axhline(
    y=rmsd, xmin=0, xmax=1, color="black", linestyle="--"
)

plt.savefig("scatter_rmsd_plddt.png")
plt.close()

# Filtering the binders based on the scores
scores_af2_filtered=scores_af2[(scores_af2['lDDT'] >=85.0) & (scores_af2['rmsd'] <= 1.5)]
utils.dump_scorefile(scores_af2_filtered, "filtered_scores.sc")


### Copying good predictions to a separate directory
os.chdir(AF2_DIR)

if len(scores_af2_filtered) > 0:
    os.makedirs("good", exist_ok=True)
    good_af2_models = [row["Output_PDB"]+".pdb" for idx,row in scores_af2_filtered.iterrows()]
    for pdb in good_af2_models:
        copy2(f"part1/{pdb}", f"good/{pdb}")
    good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb")
else:
    sys.exit("No good models to continue this pipeline with")







#---------------------------------------------------

#---------------------------------------------------
