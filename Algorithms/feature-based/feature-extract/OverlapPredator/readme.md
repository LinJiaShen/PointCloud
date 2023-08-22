# Install OverlapPredator
Python=3.8.5 Pytorch=1.7.1 CUDA=11.0 on Ubuntu22.04 GTX3090
```
conda create -n Predator python=3.8.5
conda activate Predator
git clone https://github.com/overlappredator/OverlapPredator.git
cd OverlapPredator
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch 
pip install -r requirements.txt 
cd cpp_wrappers; sh compile_wrappers.sh; cd ..
```
warring: Pytorch1.X and CUDA might not abale to support GTX4090

# Usage
In this study, our main objective is to obtain corresponding points using this algorithm, so we made some adjustments to the code accordingly.
We modified the scrupts/demo.py for customn the testing input, download the main.ipynb to see how to modify the input.
To use the different weights,
1. modify the `config.architechture` layer that fit the official layers settings in config/models.py
2. remove the train_loader (It caused some miss matching and a high vmemory cost. However, its too difficult to use the train_loader when the input point clouds are customize)
3. change the config_path to the weights you are going to use, all the config file(.yaml) is save in the configs/test/ , and we modified some detail in the yaml for the testing.
4. change the weights_path to the path you save the weights, and the weights can be downloaded by running `sh scripts/download_data_weight.sh`

**Warning:**
When you are testing the KITTI and ModelNet40 datasets, make sure to downsampled the input point clouds to avoid the `CUDA out of memory.`
Moreover, due to the OverlapPredator-ModelNet model artitecher is deeper, which makes it vmemory hungry, and hard to feed the bigger input to test the synthetic data.




# Debug Log

### 1. tensorboardX TypeError: Descriptors cannot not be created directly. downgrade the protobuf package to 3.20.x or lower.  
```pip install protobuf==3.20.*```



### 2. ImportError: /Path/to/OverlapPredator/cpp_wrappers/cpp_subsampling/grid_subsampling.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZSt28__throw_bad_array_new_lengthv  
This is caused by the low version of libstdc++.so.6 that makes the Cpp_wrappers cannot get the library.<br>

Solution:<br>
If you are using the conda environment, try this:<br>
**First, find the newest version of libstdc++.so.6 in your computer**<br>
```sudo find / -name "libstdc++.so.6```<br>
**Then, check if your newest version of the libstdc++.so.6 support the throw_bad_array**<br>
```objdump -T /usr/lib64/libstdc++.so.6.0.29  | grep throw_bad_array```<br>
**Open the conda environment of the OverlapPredator**<br>
```cd $CONDA_PREFIX/envs/your_env/lib```<br>
**Verify that `_ZSt28__throw_bad_array_new_lengthv` is missing in the target shared object**<br>
**The version of the libstdc++.so.6.0 depends on your conda python version, check it in your directory before run the code**<br>
```objdump -T libstdc++.so.6.0.26 | grep throw_bad_array```<br>

**Back-up the target shared object**<br>
```mv libstdc++.so.6.0.26 libstdc++.so.6.0.26.old```

**Change the target to point on the system's**<br>
```
rm libstdc++.so libstdc++.so.6
ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so
ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so.6
ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so.6.0.26
```

### 3. ImportError: cannot import name 'COMMON_SAFE_ASCII_CHARACTERS' from 'charset_normalizer.constant'<br>

```
pip uninstall chardet 
pip uninstall charset-normalizer 
pip install chardet 
pip install charset-normalizer==2.1.0
```
# Credit
[OverlapPredator](https://github.com/prs-eth/OverlapPredator)
