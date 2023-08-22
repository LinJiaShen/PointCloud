# Install TEASER++
Follow the [TEASER++ official guide](https://github.com/MIT-SPARK/TEASER-plusplus)

In our study, we installed the Minimal Python 3 example in Ubuntu22.04
```
sudo apt install cmake libeigen3-dev libboost-all-dev
conda create -n teaser_test python=3.6 numpy
conda activate teaser_test
conda install -c open3d-admin open3d=0.9.0.0
git clone https://github.com/MIT-SPARK/TEASER-plusplus.git
cd TEASER-plusplus && mkdir build && cd build
cmake -DTEASERPP_PYTHON_VERSION=3.6 .. && make teaserpp_python
cd python && pip install .
cd ../.. && cd examples/teaser_python_ply 
python teaser_python_ply.py
```




# Debug Log
1. **ImportError: /home/../anaconda3/envs/teaserpp/lib/python3.6/site-packages/open3d/../../../libstdc++.so.6: version `GLIBCXX_3.4.29` not found** <br>
It is caused by the low version of libstdc++, remove .. in the path to your username<br>
Solution:<br>
If you are using the conda environment, try this: <br>
First, find the newest version of libstdc++.so.6 in your computer<br>
`sudo find / -name libstdc++.so.6`<br>
Then, check if your newest version of the libstdc++.so.6 support the GLIBCXX_3.4.29<br>
```
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30  | grep GLIBCXX_3.4.29
rm /home/../anaconda3/envs/teaserpp/lib/python3.6/site-packages/open3d/../../../libstdc++.so /home/../anaconda3/envs/teaserpp/lib/python3.6/site-packages/open3d/../../../libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/../anaconda3/envs/teaserpp/lib/python3.6/site-packages/open3d/../../../libstdc++.so
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/../anaconda3/envs/teaserpp/lib/python3.6/site-packages/open3d/../../../libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30 /home/../anaconda3/envs/teaserpp/lib/python3.6/site-packages/open3d/../../../libstdc++.so.6.0.26
```
2. 
