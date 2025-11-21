"""
from the names of the binders that pass the first round of AF2 filters, get the pMPNN backbone outputs and repredict the complex strucure as a multimer with Colab Design
run relaxation
align the ligand back into the structure 
run lig MPNN without rosetta and scoring stuff on relaxed structure
(saves fasta output)
"""

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
diffusion_script = "/work/lpdi/users/eline/rf_diffusion_all_atom/run_inference.py"  # edit this
proteinMPNN_script = f"{SCRIPT_DIR}/lib/LigandMPNN/run.py"  # from submodule
AF2_script = f"{SCRIPT_DIR}/scripts/af2/af2.py"  # from submodule
CONDAPATH = "/work/lpdi/users/eline/miniconda3"  # edit this depending on where your Conda environments live
PYTHON = {
    "diffusion": f"{CONDAPATH}/envs/diffusion/bin/python",
    # "af2":"/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python",
    "af2": f"{CONDAPATH}/envs/mlfold/bin/python",
    "proteinMPNN": f"{CONDAPATH}/envs/diffusion/bin/python",
    "general": f"{CONDAPATH}/envs/diffusion/bin/python",
    "ligandMPNN": f"{CONDAPATH}/envs/ligandmpnn_env/bin/python",
    "ColabDesign": "/work/lpdi/users/mpacesa/Pipelines/miniforge3/envs/BindCraft_kuma/bin/python"
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
os.chdir(f"{AF2_DIR}/good")
good_af2_models = glob.glob(f"{AF2_DIR}/good/*.pdb") # these models only will be redesigned
DESIGN_DIR_ligMPNN_alt= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt"
os.makedirs(DESIGN_DIR_ligMPNN_alt, exist_ok=True)
DESIGN_DIR_ligMPNN_alt_af2= f"{WDIR}/3.1_design_pocket_ligandMPNN/alt/af2_reprediction"
os.makedirs(DESIGN_DIR_ligMPNN_alt_af2, exist_ok=True)

os.chdir(DESIGN_DIR_ligMPNN_alt)

# get the "good" pMPNN backbones outputs (threaded with seq)
good_pmpnn_bb=list()
for design in good_af2_models:
    sub=os.path.basename(design).split("_")
    name="_".join(sub[0:3])+"_"+sub[5]+"_"+sub[3]+".pdb"
    good_pmpnn_bb.append(name)
 #--------------------------------------------------------------------------------------------------------------------------------------------
"""---------------------------------------------------------------------- repredict and relax structure:-------------------------------------------------------------------------"""
#---------------------------------------------------------------------- 
os.chdir(DESIGN_DIR_ligMPNN_alt_af2)

commands_reprediction = []
cmds_filename_des = "commands_reprediction"
with open(cmds_filename_des, "w") as file:
    file.write("source ~/.bashrc \n BC_env \n"
    for pdb in good_pmpnn_bb: 
        commands_reprediction.append(f"{PYTHON['ColabDesign']} {SCRIPT_DIR}scripts/af2/repredict_w_templateBC.py "
                         f"--pdb {MPNN_DIR}/backbones/{pdb}  \n" )
        file.write(commands_reprediction[-1])


print("Example design command:")
print(commands_reprediction[-1])
print("Number of commands:")
print(len(commands_reprediction))


### Running design jobs with Slurm.
submit_script = "submit_reprediction.sh"
utils.create_slurm_submit_script(filename=submit_script, name="3.1_reprediction", mem="4g", 
                                 N_cores=1, gpu=True, time="70:00:00",
                                 array_commandfile=cmds_filename_des, partition="h100")

#--------------------------------------------------------------------------------------------------------------------------------------------
"""----------------------------------------------------------------------run lig MPNN on relaxed structure:-------------------------------------------------------------------------"""
#----------------------------------------------------------------------
NSTRUCT=2
from Bio.PDB import PDBParser
parser = PDBParser(QUIET=True)
commands_design = []
cmds_filename_des = "commands_design"
with open(cmds_filename_des, "w") as file:
    for pdb in good_pmpnn_bb: ### change
        structure = parser.get_structure("x", f"{MPNN_DIR}/backbones/{pdb}")### change
        model = structure[0]             
        chain = model["A"]               
        # count only standard residues
        residues = [res for res in chain.get_residues() if res.id[0] == " "]
        target_reslist=list(map(str,range(len(residues)-256+1,len(residues))))
        #print(pdb +f"native res from the target: {target_reslist[0]}-{target_reslist[-1]}")
        keep_nat=" ".join(target_reslist) # these belong to the target protein and should not be re-designed
        temperatures=" ".join(list(("0.2", "0.3")))
        distance_redesign_cutoffs = " ".join(list(("8.0", "15.0", "500.0")))
        commands_design.append(f"{PYTHON['ligandMPNN_relax']} {SCRIPT_DIR}/scripts/design/relax_n_redesign.py " ### change name of the scipt and the pdbs!!!!
                         f"--pdb {MPNN_DIR}/backbones/{pdb} --nstruct {NSTRUCT} --redesign_d_cutoff {distance_redesign_cutoffs} --target_positions {keep_nat}"
                         f" --scoring {SCRIPT_DIR}/scripts/design/scoring/FUN_scoring.py --temperature {temperatures} \n" )
        file.write(commands_design[-1])

"""test
/work/lpdi/users/eline/miniconda3/envs/ligandmpnn_env/bin/python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/ligMPNN_light_pocket_design.py --pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_1_20_1_T0.2.pdb --scoring /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/scoring/FUN_scoring.py --redesign_d_cutoff 8.0 --target_positions 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 --nstruct 5
"""
print("Example design command:")
print(commands_design[-1])
print("Number of commands:")
print(len(commands_design))

"""another test:
/work/lpdi/users/eline/miniconda3/envs/ligandmpnn_relax/bin/python /work/lpdi/users/eline/smol_binder_diffusion_pipeline/scripts/design/relax_n_redesign.py --pdb /work/lpdi/users/eline/smol_binder_diffusion_pipeline/1Z9Yout/1_proteinmpnn/backbones/t2_1_74_4_T0.2.pdb --nstruct 2 --redesign_d_cutoff 8.0 15.0 500.0 --target_positions 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 --temperature 0.2 0.3
"""
### Running design jobs with Slurm.
submit_script = "submit_design.sh"
utils.create_slurm_submit_script(filename=submit_script, name="3.1_design_pocket_ligMPNN", mem="4g", 
                                 N_cores=1, gpu=True, time="70:00:00", array=len(commands_design),
                                 array_commandfile=cmds_filename_des, partition="h100", group=75)

"""utils.create_slurm_submit_script(filename=submit_script, name="2_af2", mem="6g",
                                      N_cores=2, gpu=True, partition="h100", time="30:00:00", email=EMAIL, array=len(commands_af2),
                                      array_commandfile=cmds_filename_af2, group=25)"""

p = subprocess.Popen(['sbatch', submit_script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(output, err) = p.communicate()

