import open3d as o3d
import teaserpp_python
import numpy as np 
import copy
from helpers import *
import os
import time

directory = "/path/to/correspondences/"
files = os.listdir(directory)
VOXEL_SIZE = 0.1
for i in range(len(files)-1):
    corres_xyz0 = []
    corres_xyz1 = []
    with open(f'{directory}correspondences{i}{i+1}.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        data = line.strip().split()
        xyz0 = [float(data[0]), float(data[1]), float(data[2])]
        xyz1 = [float(data[3]), float(data[4]), float(data[5])]
        corres_xyz0.append(xyz0)
        corres_xyz1.append(xyz1)
    
    corres_xyz0 = np.array(corres_xyz0).T
    corres_xyz1 = np.array(corres_xyz1).T
    start_time = time.time()
    teaser_solver = get_teaser_solver(VOXEL_SIZE)
    teaser_solver.solve(corres_xyz0,corres_xyz1)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser,t_teaser)
    
    end_time = time.time()
    runtime = end_time - start_time
    print(runtime)
    np.savetxt(f'/path/to/runtime/result{i}{i+1}.txt', [runtime], fmt='%.6f')
    np.savetxt(f'/path/to/transformation_matrix/transformation_matrix{i}{i+1}.txt', T_teaser, fmt='%.6f')
