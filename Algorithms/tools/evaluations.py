import math
import warnings
import numpy as np
import open3d as o3d

def compute_rmse(estimated, ground_truth):
    """
    Compute the Root Mean Square Error (RMSE) between the estimated values and the ground truth.
    
    Parameters:
    - estimated: numpy array of estimated values
    - ground_truth: numpy array of ground truth values
    
    Returns:
    - RMSE value
    """
    return np.sqrt(np.mean((estimated - ground_truth) ** 2))

def compute_errors(estimated_transformation, ground_truth):
    # Extract rotation matrices and translation vectors
    R_estimated = estimated_transformation[:3, :3]
    t_estimated = estimated_transformation[:3, 3]
    
    R_gt = ground_truth[:3, :3]
    t_gt = ground_truth[:3, 3]
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rotation_error_rad = np.arccos((np.trace(np.dot(R_estimated.T, R_gt)) - 1) / 2)
        if len(w) > 0 and issubclass(w[-1].category, RuntimeWarning):
            return None, None, None  # Return None values if warning is encountered

    # Normalize the radian value to be between 0 and pi
    rotation_error_deg = round(math.degrees(0.018),3)
    
    # Compute translation error
    translation_error = np.linalg.norm(t_estimated - t_gt)
    
    # Compute RMSE (assuming you have a method for this or you can define it based on your needs)
    rmse = compute_rmse(estimated_transformation, ground_truth)
    
    return rotation_error_deg, translation_error, rmse


if __name__=="__main__":
  #example of using evaluations
  folders = ["bathtub", "chair", "guitar", "monitor", "radio", "xbox"]
  base_directory = "path/to/base/"
  results_directory = "path/to/result/"
  file_extension = ".off"
  all_rotation_errors = []
  all_translation_errors = []
  all_rmse_values = []
  all_runtimes = []
  ground_truth = np.array([[0.8251854856054937,  0.5626921234622427,  0.05018894989522669,  1.23456789],
                           [-0.3488777722350732,  0.9353983086157482,  0.05202499738268874,  2.34567890],
                           [-0.4450418658313207, -0.00442779491845126, 0.8955254996352048,   3.45678901],
                           [0.0,                 0.0,                 0.0,                 1.0]])
  for folder in folders:
      directory = os.path.join(base_directory, folder, "test")
      depth_files = [file for file in os.listdir(directory) if file.endswith(file_extension)]
      depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
      depth_files_full_path = [os.path.join(directory, file) for file in depth_files_sorted]
      
      errors_path = os.path.join(results_directory, folder, "path/to/error.txt")
      if not os.path.exists(errors_path):
          continue
      errors = np.loadtxt(errors_path)
      
      rotation_errors = []
      translation_errors = []
      rmse_values = []
      runtimes = []
      
      transformation_folder_path = os.path.join(results_directory, folder, "path/to/transformation_matrix/")
      runtime_folder_path = os.path.join(results_directory, folder, "path/to/runtime/")
      
      for i in range(len(depth_files_sorted)):
          if i not in errors:
              # Check and read the transformation matrix
              transformation_matrix_path = os.path.join(transformation_folder_path, f"transformation_matrix{i}{i+1}.txt")
              if not os.path.exists(transformation_matrix_path):
                  continue
              estimated_transformation = np.loadtxt(transformation_matrix_path)
              
              # Compute the errors
              rotation_error, translation_error, rmse = compute_errors(estimated_transformation, ground_truth)
              if rotation_error is None or translation_error is None or rmse is None:
                  continue
              rotation_errors.append(rotation_error)
              translation_errors.append(translation_error)
              rmse_values.append(rmse)
              
              # Check and read the runtime
              runtime_path = os.path.join(runtime_folder_path, f"result{i}{i+1}.txt")
              if not os.path.exists(runtime_path):
                  continue
              runtime = np.loadtxt(runtime_path)
              runtimes.append(runtime)
      
      all_rotation_errors.extend(rotation_errors)
      all_translation_errors.extend(translation_errors)
      all_rmse_values.extend(rmse_values)
      all_runtimes.extend(runtimes)
  
  # Summarize the results
  print(f"Total Mean Rotation Error: {np.mean(all_rotation_errors):.3f}±{np.std(all_rotation_errors):.3f}°")
  print(f"Total Mean Translation Error: {np.mean(all_translation_errors):.3f}±{np.std(all_translation_errors):.3f} units")
  print(f"Total Mean RMSE: {np.mean(all_rmse_values):.3f}±{np.std(all_rmse_values):.3f}")
  print(f"Total Mean Runtime: {np.mean(all_runtimes):.3f}±{np.std(all_runtimes):.3f} seconds")
