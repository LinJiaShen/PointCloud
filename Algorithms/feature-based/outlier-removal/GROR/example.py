import subprocess
import re
import numpy as np
import open3d as o3d
import os
import time

def run_cpp_program(pcd1_path, pcd2_path, resolution, n_optimal, cor1_path, cor2_path):
    
    cpp_program_path = "/home/felix/桌面/PointCloud/Algorithms/feature-based/outlier-removal/GROR/build/GrorReg"
    arguments = [pcd1_path, pcd2_path, resolution, n_optimal, cor1_path, cor2_path]
    result = subprocess.run([cpp_program_path] + arguments, capture_output=True, text=True)
    if result.returncode == 0:
        print("Script ran successfully!")
        #print("Output:")
        #print(result.stdout)
        output = result.stdout
    else:
        print("Error running the script!")
        print("Error message:")
        print(result.stderr)
    #print(result.stdout)
    output = result.stdout.splitlines()
    return output
def get_ressult(output):
    best_final_TM = None
    total_time_cost = None
    for index, line in enumerate(output):
        if line.startswith("best final TM:"):
            matrix_lines = output[index + 1: index + 5]
            matrix_values = [line.split() for line in matrix_lines]
            best_final_TM = [[float(value) for value in line] for line in matrix_values]
        elif line.startswith("/*total registration time cost:"):
            total_time_cost = float(output[index + 1])  # Get the next line's value and convert to float
            
    return best_final_TM, total_time_cost

def read_point_cloud(path):
    # Initialize an empty PointCloud for safety
    point_cloud = o3d.geometry.PointCloud()

    if path.endswith(".ply") or path.endswith(".pcd"):
        point_cloud = o3d.io.read_point_cloud(path)
    elif path.endswith(".csv"):
        point_cloud_data = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=(1, 2, 3))
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    elif path.endswith(".bin"):
        with open(path, "rb") as file:
            data = file.read()
        array = np.frombuffer(data, dtype=np.float32)
        num_points = array.shape[0] // 4
        points = np.reshape(array, (num_points, 4))[:, :3]  # Extract XYZ coordinates
        point_cloud.points = o3d.utility.Vector3dVector(points)
    elif path.endswith(".off"):
        with open(path, 'r') as f:
            lines = f.readlines()

        # Check if the format is incorrect
        if "OFF" in lines[0] and not lines[0].startswith("OFF\n"):
            # Split the line after "OFF"
            header_line = lines[0].replace("OFF", "").strip()
            corrected_lines = ["OFF\n", header_line + "\n"] + lines[1:]

            # Write the corrected content back to the file
            with open(path, 'w') as f:
                f.writelines(corrected_lines)

        mesh = o3d.io.read_triangle_mesh(path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=4000)
                
    else:
        print("Unsupported file format!")
    return point_cloud
def preprocessing(corres_idx0, corres_idx1, depth_files_full_path0, depth_files_full_path1, VOXEL_SIZE, transformation = None):
    corrs_A = np.loadtxt(corres_idx0)
    corrs_B = np.loadtxt(corres_idx1)
    corrs_A = corrs_A.flatten().astype(int)
    corrs_B = corrs_B.flatten().astype(int)
    pcdA = read_point_cloud(depth_files_full_path0)
    pcdB = read_point_cloud(depth_files_full_path1)
    if VOXEL_SIZE is not None:
        pcdA = pcdA.voxel_down_sample(VOXEL_SIZE)
        pcdB = pcdB.voxel_down_sample(VOXEL_SIZE)
    if transformation is not None:
        pcdB = pcdB.transform(transformation)
    pcdA_xyz = np.array(pcdA.points)
    pcdB_xyz = np.array(pcdB.points)
    corrs_xyzA = pcdA_xyz[corrs_A]
    corrs_xyzB = pcdB_xyz[corrs_B]
    pcdA.points = o3d.utility.Vector3dVector(corrs_xyzA)
    pcdB.points = o3d.utility.Vector3dVector(corrs_xyzB)
    corresA, corresB =  construct_correspondences(corrs_xyzA, corrs_xyzB)
    return pcdA, pcdB, corresA, corresB
def construct_correspondences(corrs_xyzA, corrs_xyzB):
    assert corrs_xyzA.shape[0] == corrs_xyzB.shape[0]
    corresA = np.array(range(corrs_xyzA.shape[0]))
    corresB = np.array(range(corrs_xyzB.shape[0]))
    return corresA, corresB


if __name__ == "__main__":
  cor_directory = "/path/to/cor_dir/"
  data_directory = "/path/to/data_dir/"
  file_extension = ".csv"
  depth_files = [file for file in os.listdir(data_directory) if file.endswith(file_extension)]
  depth_files.remove(f'icpList.csv')
  depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
  depth_files_full_path = [os.path.join(data_directory, file) for file in depth_files_sorted]
  VOXEL_SIZE = 0.1
  for i in range(len(depth_files_full_path)-1):
      pcdA, pcdB, corresA, corresB = preprocessing(f'{cor_directory}{i}{i+1}/corres_idx0.txt', f'{cor_directory}{i}{i+1}/corres_idx1.txt', depth_files_full_path[i], depth_files_full_path[i+1], VOXEL_SIZE)
      o3d.io.write_point_cloud('./tempa.pcd', pcdA, write_ascii=True)
      o3d.io.write_point_cloud('./tempb.pcd', pcdB, write_ascii=True)
      np.savetxt('corres_idx0.txt', corresA, fmt='%d')
      np.savetxt('corres_idx1.txt', corresB, fmt='%d')
      output = run_cpp_program('./tempa.pcd','./tempb.pcd',f'{VOXEL_SIZE}','800', 'corres_idx0.txt', 'corres_idx1.txt')
      best_final_TM, total_time_cost = get_ressult(output)
      output_path = '/path/to/output/'
      if not os.path.exists(output_path):
          os.makedirs(output_path)
          os.makedirs(f"{output_path}runtime/")
          os.makedirs(f"{output_path}transformation_matrix/")
      np.savetxt(f'{output_path}runtime/result{i}{i+1}.txt', [float(total_time_cost)], fmt='%.6f')
      np.savetxt(f'{output_path}transformation_matrix/transformation_matrix{i}{i+1}.txt', best_final_TM, fmt='%.6f')
