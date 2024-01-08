import os
import time
import numpy as np
import open3d as o3d

data_directory = "path/to/data/" 
file_extension = ".ply"
depth_files = [file for file in os.listdir(data_directory) if file.endswith(file_extension)]
depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
depth_files_full_path = [os.path.join(data_directory, file) for file in depth_files_sorted]
VOXEL_SIZE = 0.1
for i in range(len(depth_files_full_path-1)):
    source = o3d.io.read_point_cloud(depth_files_full_path[i])
    target = o3d.io.read_point_cloud(depth_files_full_path[i+1])
    
    # Downsample the point clouds
    source_down = source.voxel_down_sample(VOXEL_SIZE)
    target_down = target.voxel_down_sample(VOXEL_SIZE)
    
    # ICP registration
    start_time = time.time()
    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down, VOXEL_SIZE)
    icp_runtime = time.time() - start_time
    icp_transformation =icp_result.transformation
    
    # Save the transformation matrices and runtimes
    np.savetxt(f"path/to/transformation_matrix/transformation_matrix{i}{i+1}.txt", icp_transformation)
    np.savetxt(f"path/to/runtime/runtime.txt{i}{i+1}", [icp_runtime])
