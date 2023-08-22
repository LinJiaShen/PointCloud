import numpy as np
import random
# Generate point cloud 1 with four points
point_cloud_1 = np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 1],
                         [1, 1, 0]])

# Define rotation and translation for point cloud 2
rotation = np.array([[0.96592582, -0.25881905, 0],
                     [0.25881905, 0.96592582, 0],
                     [0.7863521, -0.65413282, 1]])
translation = np.array([0.5, 0.5, 0.2])
scale = 1.0
R_combined = np.vstack((scale*rotation, np.array([0, 0, 0])))

# Add an extra column to the rotation matrix
t_combined = np.append(translation, 1)

# Combine the rotation matrix and translation vector
transformation_matrix_oringin = np.column_stack((R_combined, t_combined))



point_cloud_2 = np.dot(point_cloud_1, scale*rotation)+ translation
# Find correspondences

correspondences = [(i, i) for i in range(len(point_cloud_1))]  
# Points with the same index correspond to each other


# build the set of correspondences M = {(xi, yi)}, where i is 1 to the total Number of correspondeces

M = [(point_cloud_1[i], point_cloud_2[i]) for i in range(len(correspondences))]
M = np.asarray(M) # Shape (number of corres, number of point clouds, 3)
print(M)
1/0
# construct the line vector
line_vectors_x = M[1:, 0, :] - M[:-1, 0, :]
line_vectors_y = M[1:, 1, :] - M[:-1, 1, :]

# scale estimate

# set of scale vector

line_vectors_s = [] #s_k
line_vectors_tao = [] #tai_k (set of inlier thresholds)
tao = 0.3

H = correspondences #index of the scale obeservation line_vectors_s(s_k)

for vector_x, vector_y in zip(line_vectors_x, line_vectors_y):
    
    line_vectors_s.append(np.linalg.norm(vector_y)/np.linalg.norm(vector_x))
    line_vectors_tao.append(2*tao / np.linalg.norm(vector_x))

# Set parameters
max_iter = 10000  # Maximum number of iterations 
best_s = 0  # Best estimated scale
best_numbers_of_inliers = 0  # Number of inliers for the best estimated scale
best_Is = []  # Consensus set of inliers for the best estimated scale
K = len(line_vectors_s)  # Total number of scale observations
for i in range(max_iter):
    # Randomly pick a scale s_i from the set of scale observations
    s_i = line_vectors_s[random.randint(0, K-1)]

    # Find a consensus set Is
    Is = []
    for k in range(K):
        # Check if the k-th scale observation satisfies the inlier condition
        if abs(line_vectors_s[k] - s_i) / line_vectors_tao[k] <= 1:
            Is.append(k)

    # Check if the number of inliers in the consensus set is greater than the current best
    if len(Is) > best_numbers_of_inliers:
        # estimate a refined scale s'^i accroding to (10)
        sum_inv_tao_sq = 0
        sum_s_tao_inv_tao_sq = 0
        best_Is = Is
        for k in best_Is:
            sum_inv_tao_sq += 1 / (line_vectors_tao[k] ** 2)
            sum_s_tao_inv_tao_sq += line_vectors_s[k] / (line_vectors_tao[k] ** 2)

        refined_s = sum_s_tao_inv_tao_sq / sum_inv_tao_sq
        
        #find a refind conesesus set (I'^s)_i
        refined_Is = []
        for k in range(K):
        # Check if the k-th scale observation satisfies the inlier condition
            if abs(line_vectors_s[k] - refined_s) / line_vectors_tao[k] <= 1:
                refined_Is.append(k)
        best_Is = refined_Is
        best_numbers_of_inliers = len(refined_Is)
        best_s = refined_s

# Print the best estimated scale, the refined consensus set, and the number of refined inliers
print("Best Estimated Scale:", best_s)
print("Refined Consensus Set of Inliers:", best_Is)
print("Number of Refined Inliers:", best_numbers_of_inliers)


# Rotation Estimation

# scale-annealing biweight for rotation estimation
from scipy.linalg import svd

def formulate_wls_problem(x_k, y_k, best_s, R, u, weights):
    num_correspondences = len(x_k)

    # Initialize the weighted least-squares (WLS) problem
    A = np.zeros((num_correspondences * 3, 9))
    b = np.zeros((num_correspondences * 3, 1))
    W = np.zeros((num_correspondences * 3, num_correspondences * 3))

    # Compute residuals
    residuals = np.linalg.norm(y_k - best_s * np.dot(R, x_k.T), axis=1)

    # Formulate the WLS problem
    for i in range(num_correspondences):
        A[i * 3: (i + 1) * 3, :] = np.kron(x_k[i, :], R.T)
        b[i * 3: (i + 1) * 3, 0] = y_k[i, :]
        W[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] = weights[i] * np.eye(3)

    # Solve the WLS problem
    solution = np.linalg.lstsq(np.dot(A.T, np.dot(W, A)), np.dot(A.T, np.dot(W, b)), rcond=None)

    # Reshape the solution to obtain the rotation matrix
    estimated_R = solution[0].reshape(3, 3)

    return estimated_R

def calculate_residuals(x_k, y_k, best_s, R):
    residuals = np.linalg.norm(y_k - best_s * np.dot(R, x_k.T), axis=1)
    return residuals

def update_weights(residuals, u):
    weights = np.where(np.abs(residuals) <= u, (1 - ((residuals ** 2) / u ** 2)) ** 2, 0)
    return weights

# Initialize variables and parameters
u = 1000
alpha = 1.3
w = np.ones(len(best_Is))
converged = False
# Initialize R
R = np.eye(3)
first = True
while not converged:
    if first:
        # Update weights {w(rk)} according to (14)
        residuals = calculate_residuals(line_vectors_x, line_vectors_y, best_s, R)
        w = update_weights(residuals, u)
        first = False

    # Formulate the Weighted Least Squares (WLS) problem according to (13)
    solution = formulate_wls_problem(line_vectors_x, line_vectors_y, best_s, R, u, w)
    # Estimate a solution R via the Singular Value Decomposition (SVD)
    U, _, Vt = svd(solution)
    R = np.dot(U, Vt)
    
    # Update weights {w(rk)} according to (14)
    residuals = calculate_residuals(line_vectors_x, line_vectors_y, best_s, R)
    w = update_weights(residuals, u)
    # Anneal the scale by u = u / alpha
    u = u / alpha

    # Check for convergence (e.g., based on a maximum number of iterations or other criteria)
    if u < 1:
        converged = True
        break
solution = formulate_wls_problem(line_vectors_x, line_vectors_y, best_s, R, u, w)
# Estimate a solution R via the Singular Value Decomposition (SVD)
U, _, Vt = svd(solution)
R = np.dot(U, Vt)
best_R = R
print(best_R)

# Translation Estimation

# Set parameters
max_iter = 10000  # Maximum number of iterations 
best_t = np.zeros(3) # Best estimated translation
best_numbers_of_inliers_t = 0  # Number of inliers for the best estimated scale
best_It = []  # Consensus set of inliers for the best estimated scale
K_plum = len(best_Is)  # Total number of scale observations


for i in range(max_iter):
    # Randomly select a correspondence for t_i
    # random_index = random.randint(0, K_plum - 1)
    random_index = random.randint(0, len(line_vectors_x) - 1)
    x_i = line_vectors_x[random_index]
    y_i = line_vectors_y[random_index]
    t_i = y_i - best_s * np.dot(best_R, x_i.T)

    # Calculate residuals and find inliers
    residuals = np.linalg.norm(np.array([t_i, best_t]))

    It = []
    for k in range(K_plum):
        if np.linalg.norm(np.array([line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k]), best_t])) <= 2*tao:
            It.append(k)
    # Update best inliers and number of inliers if current set is better
    if len(It) > best_numbers_of_inliers_t:
        best_inliers = It
        best_numbers_of_inliers_t = len(It)
        # refine tranlsation
        sum_t = 0
        for k in best_inliers:
            sum_t += line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k].T)
        refined_t = sum_t / best_numbers_of_inliers_t
        refined_It = []
        for k in range(K_plum):
            if np.linalg.norm(np.array([line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k]), t_i])) <= 2*tao:
                refined_It.append(k)
        best_t = refined_t
        best_inliers = refined_It
        best_numbers_of_inliers_t = len(refined_It)


R_combined = np.vstack((best_s*best_R, np.array([0, 0, 0])))

# Add an extra column to the rotation matrix
t_combined = np.append(best_t, 1)

# Combine the rotation matrix and translation vector
transformation_matrix = np.column_stack((R_combined, t_combined))

print(transformation_matrix)
print(best_numbers_of_inliers_t)




def rmse(point_cloud_1, point_cloud_2):
    # Check if the two point clouds have the same number of points
    if len(point_cloud_1) != len(point_cloud_2):
        raise ValueError("Point clouds have different sizes.")

    # Compute the squared differences between corresponding points
    squared_diff = np.sum(np.square(point_cloud_1 - point_cloud_2), axis=1)

    # Compute the mean squared error
    mse = np.mean(squared_diff)

    # Compute the root mean squared error
    rmse = np.sqrt(mse)

    return rmse


error = rmse(np.dot(point_cloud_1, scale*best_R) + best_t, point_cloud_2)

print("Root Mean Squared Error:", error)