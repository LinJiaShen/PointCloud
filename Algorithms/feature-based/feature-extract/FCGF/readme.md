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
# Install Pytorch, get the right version and install instructions from https://pytorch.org/get-started/locally/
# After installed the pytorch, install global pybind for mikowskiEngine
pip install "pybind11[global]"
# Install MinkowskiEngine
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"
```
Next, download FCGF git repository and install the requirements
```
git clone https://github.com/chrischoy/FCGF.git
cd FCGF
# Do the following inside the conda environment
pip install -r requirements.txt
```

# Usage
For testing your own data, you can refer the `main_extract` function in main.ipynb


# Debug Log
1. **ImportError: /home/../anaconda3/envs/fcgf/lib/python3.7/site-packages/MinkowskiEngineBackend/_C.cpython-37m-x86_64-linux-gnu.so: undefined symbol: _ZNSt15__exception_ptr13exception_ptr9_M_addrefEv** <br>
Replace the /home/../ to /home/your_user_name/
```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  | grep _ZNSt15__exception_ptr13exception_ptr9_M_addrefEv
rm /home/../anaconda3/envs/fcgf/lib/python3.7/site-packages/open3d/../../../libstdc++.so.6 /home/../anaconda3/envs/fcgf/lib/python3.7/site-packages/open3d/../../../libstdc++.so.6 /home/../anaconda3/envs/fcgf/lib/python3.7/site-packages/open3d/../../../libstdc++.so.6.0.26
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/../anaconda3/envs/fcgf/lib/python3.7/site-packages/open3d/../../../libstdc++.so
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/../anaconda3/envs/fcgf/lib/python3.7/site-packages/open3d/../../../libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/../anaconda3/envs/fcgf/lib/python3.7/site-packages/open3d/../../../libstdc++.so.6.0.26
```
2. 



# Credit
[FCGF](https://github.com/chrischoy/FCGF)
