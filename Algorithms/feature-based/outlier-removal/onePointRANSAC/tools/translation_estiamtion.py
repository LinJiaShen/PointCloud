import numpy as np
import random
from multiprocessing import Pool
def translation_estimation(line_vectors_x, line_vectors_y, best_s, best_R, best_Is, tao=0.3, max_iter=1000, sample_ratio=0.1):
    # Set parameters
    best_t = np.zeros(3)
    best_numbers_of_inliers_t = 0
    best_It = []
    K_plum = len(best_Is)
    best_inliers = best_Is

    for i in range(max_iter):
        num_samples = int(sample_ratio * len(best_Is))
        sample_indices = random.sample(range(len(best_Is)), num_samples)
        sample_vectors_x = line_vectors_x[best_Is][sample_indices]
        sample_vectors_y = line_vectors_y[best_Is][sample_indices]
        random_index = random.randint(0, len(sample_indices) - 1)
        sample_K_plum = len(sample_indices)
        x_i = sample_vectors_x[random_index]
        y_i = sample_vectors_y[random_index]
        t_i = y_i - best_s * np.dot(best_R, x_i.T)

        residuals = np.linalg.norm(np.array([t_i, best_t]))
        It = []
        for k in range(sample_K_plum):
            if np.linalg.norm(np.array([line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k]), best_t])) <= 2 * tao:
                It.append(k)

        if len(It) > best_numbers_of_inliers_t:
            best_inliers = It
            best_numbers_of_inliers_t = len(It)
            sum_t = 0
            for k in best_inliers:
                sum_t += line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k].T)
            refined_t = sum_t / best_numbers_of_inliers_t
            refined_It = []
            for k in range(sample_K_plum):
                if np.linalg.norm(np.array([line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k]), t_i])) <= 2 * tao:
                    refined_It.append(k)
            best_t = refined_t
            best_inliers = refined_It
            best_numbers_of_inliers_t = len(refined_It)
    print(best_t)
    return best_t, best_inliers


def process_translation_estimation(params):
    line_vectors_x, line_vectors_y, best_s, best_R, best_Is, tao, max_iter, sample_ratio = params
    
    best_t = np.zeros(3)
    best_numbers_of_inliers_t = 0
    best_It = []
    K_plum = len(best_Is)
    
    for i in range(max_iter):
        num_samples = int(sample_ratio * len(best_Is))
        sample_indices = random.sample(range(len(best_Is)), num_samples)
        sample_vectors_x = line_vectors_x[best_Is][sample_indices]
        sample_vectors_y = line_vectors_y[best_Is][sample_indices]
        random_index = random.randint(0, len(sample_indices) - 1)
        sample_K_plum = len(sample_indices)
        x_i = sample_vectors_x[random_index]
        y_i = sample_vectors_y[random_index]
        t_i = y_i - best_s * np.dot(best_R, x_i.T)
        
        residuals = np.linalg.norm(np.array([t_i, best_t]))
        It = []
        for k in range(sample_K_plum):
            if np.linalg.norm(np.array([line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k]), best_t])) <= 2 * tao:
                It.append(k)
        
        if len(It) > best_numbers_of_inliers_t:
            best_inliers = It
            best_numbers_of_inliers_t = len(It)
            sum_t = 0
            for k in best_inliers:
                sum_t += line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k].T)
            refined_t = sum_t / best_numbers_of_inliers_t
            refined_It = []
            for k in range(sample_K_plum):
                if np.linalg.norm(np.array([line_vectors_y[k] - best_s * np.dot(best_R, line_vectors_x[k]), t_i])) <= 2 * tao:
                    refined_It.append(k)
            best_t = refined_t
            best_inliers = refined_It
            best_numbers_of_inliers_t = len(refined_It)
    
    return best_t, best_inliers

def translation_estimation_m(line_vectors_x, line_vectors_y, best_s, best_R, best_Is, tao=0.3, max_iter=1000, sample_ratio=0.1):
    # Create a list of parameters for each process
    params = [(line_vectors_x, line_vectors_y, best_s, best_R, best_Is, tao, max_iter, sample_ratio)] * max_iter
    
    # Perform translation estimation iterations using multiprocessing
    with Pool() as pool:
        results = pool.map(process_translation_estimation, params)

    # Select the best translation estimation
    best_t, best_inliers = max(results, key=lambda x: len(x[1]))

    return best_t, best_inliers