





# Debug Log
Predator Install
##1. tensorboardX TypeError: Descriptors cannot not be created directly. downgrade the protobuf package to 3.20.x or lower.
pip install protobuf==3.20.*



##2. ImportError: /Path/to/OverlapPredator/cpp_wrappers/cpp_subsampling/grid_subsampling.cpython-38-x86_64-linux-gnu.so: undefined symbol: _ZSt28__throw_bad_array_new_lengthv
This is caused by the low version of libstdc++.so.6 that makes the Cpp_wrappers cannot get the library.

Solution:
If you are using the conda environment, try this:
**First, find the newest version of libstdc++.so.6 in your computer**
`sudo find / -name "libstdc++.so.6`
**Then, check if your newest version of the libstdc++.so.6 support the throw_bad_array**
`objdump -T /usr/lib64/libstdc++.so.6.0.29  | grep throw_bad_array`
**Open the conda environment of the OverlapPredator**
`cd $CONDA_PREFIX/envs/your_env/lib`
**Verify that `_ZSt28__throw_bad_array_new_lengthv` is missing in the target shared object**
**The version of the libstdc++.so.6.0 depends on your conda python version, check it in your directory before run the code**
`objdump -T libstdc++.so.6.0.26 | grep throw_bad_array`

**Back-up the target shared object**
`mv libstdc++.so.6.0.26 libstdc++.so.6.0.26.old`

**Change the target to point on the system's**
`rm libstdc++.so libstdc++.so.6
ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so
ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so.6
ln -s /usr/lib64/libstdc++.so.6.0.30 libstdc++.so.6.0.26`
