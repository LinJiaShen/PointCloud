'''
Inputs: 

3D correspondeces: M (an nd numpy array that saves pairs of the coresspondence points)

error torlent: tao

'''
import sys
sys.path.append("./tools/")
import tools.scale_estimation as se
import tools.rotation_estimation as re
import tools.translation_estiamtion as te
import numpy as np
# 1. line vectorize

def onepointRANSAC(M, tao, estimate_scale = True):
    # construct the line vector
    line_vectors_x = M[1:, 0, :] - M[:-1, 0, :] # line vector set from correspondeced point cloud x
    line_vectors_y = M[1:, 1, :] - M[:-1, 1, :] # line vector set from correspondeced point cloud y

    # line_vectors_x, line_vectors_y = calculate_line_vectors(point_cloud_1, point_cloud_2)
    
    
    if estimate_scale:
        best_s, best_Is, _ =se.scale_estimation(line_vectors_x, line_vectors_y, tao, max_iter=10000)
        print("Best Estimated Scale:", best_s)
    else: 
        _, best_Is, _ =se.scale_estimation(line_vectors_x, line_vectors_y, tao, max_iter=10000)
        best_s = 1

    best_R = re.rotation_estimation(line_vectors_x, line_vectors_y, best_s, best_Is, alpha=1.3, u=1000)
    best_t, best_It = te.translation_estimation(line_vectors_x, line_vectors_y,  best_s, best_R, best_Is, tao, max_iter=10000)
    
    R_combined = np.vstack((best_s*best_R, np.array([0, 0, 0])))

    # Add an extra column to the rotation matrix
    t_combined = np.append(best_t, 1)

    # Combine the rotation matrix and translation vector
    transformation_matrix = np.column_stack((R_combined, t_combined))

    return transformation_matrix, best_It