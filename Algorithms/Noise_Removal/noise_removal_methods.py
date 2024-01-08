import numpy as np
import open3d as o3d


def statiscal_noise_remove(pcd, nb_neighbors=20, std_ratio=2.0, visualize=True):
    """
    Filters a point cloud using statistical outlier removal and optionally visualizes the result.
    
    Parameters:
    - pcd: open3d point cloud 
    - nb_neighbors: Number of neighbors to use for the filter.
    - std_ratio: Standard deviation ratio for the filter.
    - visualize: Whether to visualize the result.
    
    Returns:
    - combined_pcd: Combined point cloud with original points
    """
    # Filter the point cloud using statistical outlier removal
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    # Visualize the result if requested
    if visualize:
        # Get the points that were removed
        removed_pcd = select_by_index(pcd, ind, invert=True)
        # Color the original points in blue
        cl.paint_uniform_color([0, 0, 1])
        # Color the removed points in red
        removed_pcd.paint_uniform_color([1, 0, 0])
        # Combine the two point clouds for visualization
        combined_pcd = cl + removed_pcd
        o3d.visualization.draw_geometries([combined_pcd])
    
    return cl, ind


def radius_noise_remove(pcd, nb_neighbors=20, radius=2.0, visualize=True):
    """
    Filters a point cloud using statistical outlier removal and optionally visualizes the result.
    
    Parameters:
    - pcd: open3d point cloud 
    - nb_neighbors: Number of neighbors to use for the filter.
    - std_ratio: Standard deviation ratio for the filter.
    - visualize: Whether to visualize the result.
    
    Returns:
    - combined_pcd: Combined point cloud with original points
    """

    # Filter the point cloud using statistical outlier removal
    cl, ind = pcd.remove_radius_outlier(nb_points=nb_neighbors, radius=radius)
    # Visualize the result if requested
    if visualize:
        # Get the points that were removed
        removed_pcd = select_by_index(pcd, ind, invert=True)
        # Color the original points in blue
        cl.paint_uniform_color([0, 0, 1])
        # Color the removed points in red
        removed_pcd.paint_uniform_color([1, 0, 0])
        # Combine the two point clouds for visualization
        combined_pcd = cl + removed_pcd
        o3d.visualization.draw_geometries([combined_pcd])
    
    return cl, ind
