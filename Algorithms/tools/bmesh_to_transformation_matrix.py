import numpy as np
from scipy.spatial.transform import Rotation


# Define the target position in 7D space (x, y, z, quatX, quatY, quatZ, quatW)
target_str = "-0.00064470991 -4.8473477e-05 0.001028873 -0.00071992492 0.86697686 -0.00451387 -0.49832746"

# Parse the target values from the string
target_vals = [float(val) for val in target_str.split()]

# Create the target transformation matrix from the target position
# Extract the translation vector from the target position values
target_trans = np.array(target_vals[:3]).reshape(3, 1)
# Extract the quaternion rotation vector from the target position values
target_quat = np.array(target_vals[3:])
# Convert the quaternion rotation vector to a rotation matrix using scipy.spatial.transform.Rotation
# Then, take the inverse of the rotation matrix as we want to go from the target to the object
target_rot =  np.linalg.inv(Rotation.from_quat(target_quat).as_matrix())
# Combine the rotation and translation vectors into a 4x4 transformation matrix
target_mat4x4 = np.hstack((target_rot, target_trans))
target_mat4x4 = np.vstack((target_mat4x4, np.array([0, 0, 0, 1])))

# Print and save the transformation matrix from the target to the object
print(target_mat4x4)
np.savetxt('E:/user/Desktop/Project/Pointcloudregistration/Datasets/Single_Object/happy/gt/transformationc5.txt', target_mat4x4)
