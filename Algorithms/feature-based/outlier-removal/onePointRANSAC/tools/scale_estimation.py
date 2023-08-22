import numpy as np
import random
from multiprocessing import Pool
def scale_estimation(line_vectors_x, line_vectors_y, tao, max_iter):
    # Initialize scale parameters and variables
    
    line_vectors_s = np.linalg.norm(line_vectors_y, axis=1) / np.linalg.norm(line_vectors_x, axis=1)

    line_vectors_tao = 2 * tao / np.linalg.norm(line_vectors_x, axis=1)
    best_s = 0
    best_numbers_of_inliers = 0
    best_Is = []
    K = len(line_vectors_s)

    # Perform scale estimation iterations
    for i in range(max_iter):
        # Randomly pick a scale s_i from the set of scale observations
        s_i = line_vectors_s[random.randint(0, K - 1)]

        # Find a consensus set Is
        Is = []
        for k in range(K):
            # Check if the k-th scale observation satisfies the inlier condition
            if abs(line_vectors_s[k] - s_i) / line_vectors_tao[k] <= 1:
                Is.append(k)

        # Check if the number of inliers in the consensus set is greater than the current best
        if len(Is) > best_numbers_of_inliers:
            # Estimate a refined scale s'^i accroding to (10)
            sum_inv_tao_sq = 0
            sum_s_tao_inv_tao_sq = 0
            best_Is = Is
            for k in best_Is:
                sum_inv_tao_sq += 1 / (line_vectors_tao[k] ** 2)
                sum_s_tao_inv_tao_sq += line_vectors_s[k] / (line_vectors_tao[k] ** 2)
            refined_s = sum_s_tao_inv_tao_sq / sum_inv_tao_sq
            # Find a refined consensus set (I'^s)_i
            refined_Is = []
            for k in range(K):
                # Check if the k-th scale observation satisfies the inlier condition
                if abs(line_vectors_s[k] - refined_s) / line_vectors_tao[k] <= 1:
                    refined_Is.append(k)

            best_Is = refined_Is
            best_numbers_of_inliers = len(refined_Is)
            best_s = refined_s
    print(best_s)

    
    return best_s, best_Is, best_numbers_of_inliers

def process_best_scale_estimation(params):
    Is, num_inliers, line_vectors_s, line_vectors_tao = params
    sum_inv_tao_sq = 0
    sum_s_tao_inv_tao_sq = 0
    best_Is = Is
    for k in best_Is:
        sum_inv_tao_sq += 1 / (line_vectors_tao[k] ** 2)
        sum_s_tao_inv_tao_sq += line_vectors_s[k] / (line_vectors_tao[k] ** 2)

    refined_s = sum_s_tao_inv_tao_sq / sum_inv_tao_sq

    # Find a refined consensus set (I'^s)_i
    refined_Is = []
    for k in range(len(line_vectors_s)):
        # Check if the k-th scale observation satisfies the inlier condition
        if abs(line_vectors_s[k] - refined_s) / line_vectors_tao[k] <= 1:
            refined_Is.append(k)

    return refined_s, refined_Is, len(refined_Is)

def process_scale_estimation(params):
    K, line_vectors_s, line_vectors_tao = params
    # Randomly pick a scale s_i from the set of scale observations
    s_i = line_vectors_s[random.randint(0, K - 1)]

    # Find a consensus set Is
    Is = []
    for k in range(K):
        # Check if the k-th scale observation satisfies the inlier condition
        if abs(line_vectors_s[k] - s_i) / line_vectors_tao[k] <= 1:
            Is.append(k)

    return Is, len(Is), s_i

def scale_estimation_m(line_vectors_x, line_vectors_y, tao, max_iter):
    # Initialize scale parameters and variables
    line_vectors_s = np.linalg.norm(line_vectors_y, axis=1) / np.linalg.norm(line_vectors_x, axis=1)
    line_vectors_tao = 2 * tao / np.linalg.norm(line_vectors_x, axis=1)
    best_s = 0
    best_numbers_of_inliers = 0
    best_Is = []
    K = len(line_vectors_s)


    params_s = [(K, line_vectors_s, line_vectors_tao)]
    # Perform scale estimation iterations using multiprocessing
    with Pool() as pool:
        results = pool.map(process_scale_estimation, params_s)

    # Check the results and find the best scale estimation
    params = [(Is, num_inliers, line_vectors_s, line_vectors_tao) for Is, num_inliers, _ in results]
    with Pool() as pool:
        refined_results = pool.map(process_best_scale_estimation, params)

    # Find the best refined scale estimation
    for refined_s, refined_Is, num_inliers in refined_results:
        if num_inliers > best_numbers_of_inliers:
            best_s = refined_s
            best_Is = refined_Is
            best_numbers_of_inliers = num_inliers

    return best_s, best_Is, best_numbers_of_inliers