# Install FCGF
We adjusted some steps from [offcial install guide](https://github.com/chrischoy/FCGF), and installed it in Ubuntu22.04 with conda.
Before the installation, please make sure you have already installed the required components.
1. Ubuntu 14.04 or higher
2. CUDA 11.1 or higher
3. Python v3.7 or higher

**Warning: The MinkowskiEngine do not support Windows, please make sure that you are install under the Linux system.** <br>
```
conda create -n py3-fcgf python=3.7
conda activate py3-fcgf
conda install numpy
conda install openblas-devel -c anaconda
conda install pytorch=1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
pip install "pybind11[global]"
# Install MinkowskiEngine
# Uncomment the following line to specify the cuda home. Make sure `$CUDA_HOME/nvcc --version` is 11.X
# export CUDA_HOME=/usr/local/cuda-11.1
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```
Next, download FCGF git repository and install the requirements
```
git clone https://github.com/chrischoy/FCGF.git
cd FCGF
# Do the following inside the conda environment
pip install -r requirements.txt
```
