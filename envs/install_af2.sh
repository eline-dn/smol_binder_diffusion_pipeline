conda create -n alphafold python=3.11 -y
conda activate alphafold

conda install -c nvidia cuda=12.2.2 -y

PATH="/opt/conda/bin:$PATH"
LD_LIBRARY_PATH="/opt/conda/lib:$LD_LIBRARY_PATH"
CONDA_PLUGINS_AUTO_ACCEPT_TOS="yes"
conda install --channel nvidia cuda=12.2.2 \
    && conda install --channel conda-forge openmm=8.0.0 pdbfixer \
    && conda clean --all --force-pkgs-dirs --yes

pip3 install --upgrade pip --no-cache-dir \
    && pip3 install --no-cache-dir \
        absl-py==1.0.0 \
        biopython==1.79 \
        dm-haiku==0.0.12 \
        docker==5.0.0 \
        matplotlib==3.8.0 \
        jax==0.4.26 \
        ml-collections==0.1.0 \
        numpy==1.24.3 \
        "pytest<8.5.0" \
        "setuptools<72.0.0" \
        tensorflow-cpu==2.16.1 \
    && pip3 install --upgrade --no-cache-dir \
        jaxlib==0.4.26+cuda12.cudnn89 \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
#chmod u+s /sbin/ldconfig.real
# Currently needed to avoid undefined_symbol error.
#ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 /opt/conda/lib/libffi.so.7
pip install mock
pip install dm-tree