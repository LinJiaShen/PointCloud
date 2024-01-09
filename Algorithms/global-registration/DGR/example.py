import numpy as np
import os
from urllib.request import urlretrieve
import open3d as o3d
from core.deep_global_registration import DeepGlobalRegistration
import time
import subprocess
from config import get_config
import sys
sys.path.append('.')
from core.registration import GlobalRegistration
import torch
import re

def run_program(pcd1_path, pcd2_path, model):
    
    program_path = "demo.py"
    
    arguments = ['--pcd0', pcd1_path, '--pcd1', pcd2_path, '--weights', model]
    
    #result = subprocess.run([f"demo.py --pcd0 {pcd1_path} --pcd1 {pcd2_path} --model {model}"], capture_output=True)
    result = subprocess.run(["python", program_path]+arguments, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Script ran successfully!")
        #print("Output:")
        #print(result.stdout)
        output = result.stdout
    else:
        print("Error running the script!")
        print("Error message:")
        print(result.stderr)
    output = result.stdout.splitlines()
    return output

directory = "/path/to/data/"  # Replace with the actual directory path
file_extension = ".csv"
model = "ResUNetBN2C-feat32-3dmatch-v0.05.pth"
depth_files = [file for file in os.listdir(directory) if file.endswith(file_extension)]
depth_files.remove(f'icpList.csv')
depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
depth_files_full_path = [os.path.join(directory, file) for file in depth_files_sorted]
output_path = '/path/to/output/'
print(depth_files_full_path)
VOXEL_SIZE = 0.1
for i in range(len(depth_files_full_path)-1):
    
    A_pcd_raw = read_point_cloud(f"{depth_files_full_path[i]}")
    B_pcd_raw = read_point_cloud(f"{depth_files_full_path[i+1]}")
    A_pcd = A_pcd_raw.voxel_down_sample(VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(VOXEL_SIZE)
    o3d.io.write_point_cloud('./tempa.pcd', A_pcd, write_ascii=True)
    o3d.io.write_point_cloud('./tempb.pcd', B_pcd, write_ascii=True)
    start_time = time.time()
    output = run_program('tempa.pcd', 'tempb.pcd', model)
    end_time = time.time()
    
    for idx, line in enumerate(output):
        if "Transformation" in line:
            matrix_lines = output[idx + 1 : idx + 5]
            break
    trans = []
    for lines in matrix_lines :
        # Use regular expression to extract the transformation matrix values from the line
        matrix_values = re.findall(r"[-+]?\d*\.\d+|\d+", lines)
        # Convert the list of values to a 4x4 NumPy array
        trans.append(matrix_values)
    transformation_matrix = np.array(trans, dtype=float).reshape(4, 4)
    
    
    runtime = end_time - start_time
    if not os.path.exists(output_path):
        os.makedirs(f"{output_path}runtime/")
        os.makedirs(f"{output_path}transformation_matrix/")
    np.savetxt(f'{output_path}runtime/result{i}{i+1}.txt', [float(runtime)], fmt='%.6f')
    np.savetxt(f'{output_path}transformation_matrix/transformation_matrix{i}{i+1}.txt', transformation_matrix, fmt='%.6f')
    
