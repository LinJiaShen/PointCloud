import numpy as np

def calculate_line_vectors(point_cloud_1, point_cloud_2):
    line_vectors_x = point_cloud_1[1:, :] - point_cloud_1[:-1, :]
    line_vectors_y = point_cloud_2[1:, :] - point_cloud_2[:-1, :]
    return line_vectors_x, line_vectors_y