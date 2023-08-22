from scipy.linalg import svd
import numpy as np
import random
from multiprocessing import Pool
def rotation_estimation(line_vectors_x, line_vectors_y, best_s, best_Is, alpha=1.3, u=1000, sample_ratio=0.1):
    # Initialize variables and parameters
    w = np.ones(len(best_Is))
    converged = False
    R = np.eye(3)
    first = True

    def formulate_wls_problem(x_k, y_k, R, weights):
        num_correspondences = len(x_k)

        # Initialize the weighted least-squares (WLS) problem
        A = np.zeros((num_correspondences * 3, 9))
        b = np.zeros((num_correspondences * 3, 1))
        W = np.zeros((num_correspondences * 3, num_correspondences * 3))

        # Compute residuals
        residuals = calculate_residuals(x_k, y_k, R)

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

    def calculate_residuals(x_k, y_k, R):
        residuals = np.linalg.norm(y_k - (best_s * np.dot(R, x_k.T)).T, axis=1)
        return residuals

    def update_weights(residuals, u):
        weights = np.where(np.abs(residuals) <= u, (1 - ((residuals ** 2) / u ** 2)) ** 2, 0)
        return weights
    
    while not converged:
        # Randomly sample a subset of inliers for WLS estimation
        #num_samples = int(sample_ratio * len(best_Is))
        #sample_indices = random.sample(range(len(best_Is)), num_samples)
        #sample_vectors_x = line_vectors_x[best_Is][sample_indices]
        #sample_vectors_y = line_vectors_y[best_Is][sample_indices]
        sample_vectors_x = line_vectors_x[best_Is]
        sample_vectors_y = line_vectors_y[best_Is]
        if first:
            # Update weights {w(rk)} according to (14)
            residuals = calculate_residuals(sample_vectors_x, sample_vectors_y, R)
            w = update_weights(residuals, u)
            first = False

        # Formulate the Weighted Least Squares (WLS) problem according to (13)
        solution = formulate_wls_problem(sample_vectors_x, sample_vectors_y, R, w)

        # Estimate a solution R via the Singular Value Decomposition (SVD)
        U, _, Vt = svd(solution)
        R = np.dot(U, Vt)

        # Update weights {w(rk)} according to (14)
        residuals = calculate_residuals(sample_vectors_x, sample_vectors_y, R)
        w = update_weights(residuals, u)

        # Anneal the scale by u = u / alpha
        u = u / alpha

        # Check for convergence
        if u < 1:
            converged = True
            break
    print(R)
    return R

def process_rotation_estimation(params):
    line_vectors_x, line_vectors_y, best_s, best_Is, alpha, u, max_iter, sample_ratio = params
    
    w = np.ones(len(best_Is))
    converged = False
    R = np.eye(3)
    first = True
    
    def formulate_wls_problem(x_k, y_k, R, weights):
        num_correspondences = len(x_k)

        # Initialize the weighted least-squares (WLS) problem
        A = np.zeros((num_correspondences * 3, 9))
        b = np.zeros((num_correspondences * 3, 1))
        W = np.zeros((num_correspondences * 3, num_correspondences * 3))

        # Compute residuals
        residuals = calculate_residuals(x_k, y_k, R)

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
        
    def calculate_residuals(x_k, y_k, R):
        residuals = np.linalg.norm(y_k - (best_s * np.dot(R, x_k.T)).T, axis=1)
        return residuals
        
    def update_weights(residuals, u):
        weights = np.where(np.abs(residuals) <= u, (1 - ((residuals ** 2) / u ** 2)) ** 2, 0)
        return weights

    while not converged:
        num_samples = int(sample_ratio * len(best_Is))
        sample_indices = random.sample(range(len(best_Is)), num_samples)
        sample_vectors_x = line_vectors_x[best_Is][sample_indices]
        sample_vectors_y = line_vectors_y[best_Is][sample_indices]
        
        if first:
            residuals = calculate_residuals(sample_vectors_x, sample_vectors_y, R)
            w = update_weights(residuals, u)
            first = False
        
        solution = formulate_wls_problem(sample_vectors_x, sample_vectors_y, R, w)
        U, _, Vt = svd(solution)
        R = np.dot(U, Vt)
        
        residuals = calculate_residuals(sample_vectors_x, sample_vectors_y, R)
        w = update_weights(residuals, u)
        
        u = u / alpha
        
        if u < 1:
            converged = True
            break
    
    return R

def rotation_estimation_m(line_vectors_x, line_vectors_y, best_s, best_Is, alpha=1.3, u=1000, max_iter=1000, sample_ratio=0.05):
    # Create a list of parameters for each process
    params = [(line_vectors_x, line_vectors_y, best_s, best_Is, alpha, u, max_iter, sample_ratio)] * max_iter
    
    # Perform rotation estimation iterations using multiprocessing
    with Pool() as pool:
        results = pool.map(process_rotation_estimation, params)

    # Select the best rotation estimation
    best_R = max(results, key=lambda R: np.linalg.det(R))

    return best_R