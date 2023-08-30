


# Usage
Download the model from https://github.com/qinzheng93/GeoTransformer/releases, and take off the .tar behind the model names.
Modified the demo.py, it set a default neighbor limit, which is not for all kinds of scenes.




Warning!
The algorithm is vgpu hungry, while using the indoor model and modelnet model.
To test the modelnet data, we suggest to seperate the input point clouds(reduce the amount of the points).

# Error
1. RuntimeError: CUDA out of memory.
   It is due to the vgpu hungry.
2. RuntimeError: shape '[1, 128, 256]' is invalid for input of size 0
   It is due to the algorithm cannot find the correspondence between the clouds.
