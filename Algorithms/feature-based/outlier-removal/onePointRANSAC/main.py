import open3d as o3d
import numpy as np
import sys
sys.path.append("./tools/")
from tools import get_corr as cor
import onepointRANSAC as opr
import time

datanames = ['Hokuyo_0.csv','Hokuyo_1.csv','Hokuyo_2.csv','Hokuyo_3.csv','Hokuyo_4.csv','Hokuyo_5.csv']

for j in range(len(datanames)-1):
    
    # Load point clouds
    A_pcd_raw = cor.read_point_cloud(f"E:/POINTCLOUD/Datasets/Indoor/apartment/local/{datanames[j]}")
    B_pcd_raw = cor.read_point_cloud(f"E:/POINTCLOUD/Datasets/Indoor/apartment/local/{datanames[j+1]}")

    A_corr, B_corr = cor.fpfh(A_pcd_raw, B_pcd_raw, VOXEL_SIZE= 0.1, show= False)
    num_corrs = A_corr.shape[1]
    # Construct the set of correspondences M
    M = np.zeros((num_corrs, 2, 3))  # Shape of M: (number of correspondences, number of point clouds, 3)
    for i in range(num_corrs):
        M[i, 0] = A_corr[:, i]  # Corresponding point from point cloud A
        M[i, 1] = B_corr[:, i]  # Corresponding point from point cloud B
        
    start_time = time.time()

    tao = 0.1
    transformation_matrix, best_It = opr.onepointRANSAC(M, tao, False)

    end_time = time.time()
    runtime = end_time - start_time
    print(runtime)
    np.savetxt(f'path/to/runtime/result{j}{j+1}.txt', [runtime], fmt='%.6f')
    np.savetxt(f'path/to/transformation_matrix/transformation_matrix{j}{j+1}.txt', transformation_matrix, fmt='%.6f')
