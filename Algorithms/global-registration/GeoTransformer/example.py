import numpy as np
import open3d as o3d
import os
import torch
import time
import subprocess
import re
import time

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
        point_cloud = mesh.sample_points_uniformly(number_of_points=2000)
    else:
        print("Unsupported file format!")
    return point_cloud


def run_indoor_program(pcd1_path, pcd2_path, model):
    
    program_path = "./experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/demo.py"
    
    arguments = ['--src_file', pcd1_path, '--ref_file', pcd2_path, '--weights', model]
    
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
    print(result.stdout)
    output = result.stdout.splitlines()
    return output

def run_outdoor_program(pcd1_path, pcd2_path, model):
    
    program_path = "./experiments/geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn/demo.py"
    
    arguments = ['--src_file', pcd1_path, '--ref_file', pcd2_path, '--weights', model]
    
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
    print(result.stdout)
    output = result.stdout.splitlines()
    return output

def run_single_program(pcd1_path, pcd2_path, model):
    
    program_path = "./experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn/demo.py"
    
    arguments = ['--src_file', pcd1_path, '--ref_file', pcd2_path, '--weights', model]
    
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
    print(result.stdout)
    output = result.stdout.splitlines()
    return output


if __name__=="__main__":
  directory = "/path/to/dir/"  # Replace with the actual directory path
  file_extension = ".csv"
  model = "./models/geotransformer-3dmatch.pth"
  depth_files = [file for file in os.listdir(directory) if file.endswith(file_extension)]
  depth_files.remove(f'icpList.csv')
  depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
  depth_files_full_path = [os.path.join(directory, file) for file in depth_files_sorted]
  print(depth_files_full_path)
  VOXEL_SIZE = 0.5
  error = []
  for i in range(len(depth_files_full_path)-1):
      print(i)
      pcd1 = read_point_cloud(depth_files_full_path[i])
      pcd2 = read_point_cloud(depth_files_full_path[i+1])
      pcd1 = pcd1.voxel_down_sample(VOXEL_SIZE)
      pcd2 = pcd2.voxel_down_sample(VOXEL_SIZE)
      np.save('pcd1.npy', np.array(pcd1.points))
      np.save('pcd2.npy', np.array(pcd2.points))
      try:
          start_time = time.time()
          output = run_program('./pcd1.npy', './pcd2.npy', model)
          end_time = time.time()
          runtime = end_time-start_time
          trans = []
          for lines in output :
              # Use regular expression to extract the transformation matrix values from the line
              matrix_values = re.findall(r"[-+]?\d+\.\d+[eE][-+]?\d+|[-+]?\d+\.\d+|\d+", lines)
              # Convert the list of values to a 4x4 NumPy array
              trans.append(matrix_values)
          transformation_matrix = np.array(trans, dtype=float).reshape(4, 4)
      except:
          error.append(i)
          continue
  
      output_path = '/path/to/output/'
      if not os.path.exists(output_path):
          os.makedirs(output_path)
          os.makedirs(f"{output_path}runtime/")
          os.makedirs(f"{output_path}transformation_matrix/")
      
      np.savetxt(f'{output_path}runtime/result{i}{i+1}.txt', [float(runtime)], fmt='%.6f')
      np.savetxt(f'{output_path}transformation_matrix/transformation_matrix{i}{i+1}.txt', transformation_matrix, fmt='%.6f')
      np.savetxt(f'{output_path}error.txt', error, fmt='%d')
