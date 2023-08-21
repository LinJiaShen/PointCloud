# Install OverlapPredator

`conda create -n Predator python=3.8.5` <br>
`conda activate Predator` <br>
`git clone https://github.com/overlappredator/OverlapPredator.git` <br>
`cd OverlapPredator` <br>
`conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch` <br>
`pip install -r requirements.txt` <br>
`cd cpp_wrappers; sh compile_wrappers.sh; cd ..` <br>


# Debug Log

### 1. tensorboardX TypeError: Descriptors cannot not be created directly. downgrade the protobuf package to 3.20.x or lower.  
`pip install protobuf==3.20.*`<br>



### 2. ImportError: /Path/to/OverlapPredator/cpp_wrappers/cpp_subsampling/grid_subsampling.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZSt28__throw_bad_array_new_lengthv  
This is caused by the low version of libstdc++.so.6 that makes the Cpp_wrappers cannot get the library.<br>

Solution:<br>
If you are using the conda environment, try this:<br>
**First, find the newest version of libstdc++.so.6 in your computer**<br>
`sudo find / -name "libstdc++.so.6`<br>
**Then, check if your newest version of the libstdc++.so.6 support the throw_bad_array**<br>
`objdump -T /usr/lib64/libstdc++.so.6.0.29  | grep throw_bad_array`<br>
**Open the conda environment of the OverlapPredator**<br>
`cd $CONDA_PREFIX/envs/your_env/lib`<br>
**Verify that `_ZSt28__throw_bad_array_new_lengthv` is missing in the target shared object**<br>
**The version of the libstdc++.so.6.0 depends on your conda python version, check it in your directory before run the code**<br>
`objdump -T libstdc++.so.6.0.26 | grep throw_bad_array`<br>

**Back-up the target shared object**<br>
`mv libstdc++.so.6.0.26 libstdc++.so.6.0.26.old`<br>

**Change the target to point on the system's**<br>
`rm libstdc++.so libstdc++.so.6`<br>
`ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so`<br>
`ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so.6`<br>
`ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so.6.0.26`<br>

### 3. ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'
`pip uninstall chardet `<br>
`pip uninstall charset-normalizer`<br>
`pip install chardet `<br>
`pip install charset-normalizer==2.1.0`<br>
