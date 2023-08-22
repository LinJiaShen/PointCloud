import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import multiprocessing
from functools import partial
from PIL import Image

def read_point_cloud(path):
    if path.endswith(".ply") or path.endswith(".pcd"):
        point_cloud = o3d.io.read_point_cloud(path)
    elif path.endswith(".csv"):
        point_cloud_data = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=(1, 2, 3))
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    elif path.endswith(".bin"):
        with open(path, "rb") as file:
            data = file.read()

        # Assuming the data consists of 32-bit floats
        array = np.frombuffer(data, dtype=np.float32)
        num_points = array.shape[0] // 4

        # Create an empty PointCloud
        point_cloud = o3d.geometry.PointCloud()

        # Assign XYZ coordinates to the PointCloud
        points = np.reshape(array, (num_points, 4))[:, :3]  # Extract XYZ coordinates
        point_cloud.points = o3d.utility.Vector3dVector(points)
    elif path.endswith(".png"):
        # Load the depth image
        depth_image = Image.open(path)  # Replace with the actual file path

        # Convert the depth image to a numpy array
        depth_array = np.array(depth_image)

        # Process the depth array as needed
        print(depth_array)
        1/0
    else:
        # DEBUG: for the wrong files, return a empty list.
        return point_cloud
    return point_cloud

def fpfh(A_pcd_raw, B_pcd_raw, VOXEL_SIZE, show= False):
    # voxel downsample both clouds
    A_pcd = A_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)
    B_pcd = B_pcd_raw.voxel_down_sample(voxel_size=VOXEL_SIZE)

    # extract the coordinates of both clouds as numpy array
    A_xyz = pcd2xyz(A_pcd) # np array of size 3 by N
    B_xyz = pcd2xyz(B_pcd) # np array of size 3 by M
    # extract FPFH features
    A_feats = extract_fpfh(A_pcd,VOXEL_SIZE)
    B_feats = extract_fpfh(B_pcd,VOXEL_SIZE)
    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences_parallel(
        A_feats, B_feats, mutual_filter=False)
    A_corr = A_xyz[:,corrs_A] # np array of size 3 by num_corrs
    B_corr = B_xyz[:,corrs_B] # np array of size 3 by num_corrs
    num_corrs = A_corr.shape[1]
    print(f'FPFH generates {num_corrs} putative correspondences.')
    
    if show:
        corres_visualize(A_corr,B_corr,A_pcd,B_pcd)
    return A_corr, B_corr


def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
  feat1tree = cKDTree(feat1)
  dists, nn_inds = feat1tree.query(feat0, k=knn)
  if return_distance:
    return nn_inds, dists
  else:
    return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
  nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
  corres01_idx0 = np.arange(len(nns01))
  corres01_idx1 = nns01

  if not mutual_filter:
    return corres01_idx0, corres01_idx1

  nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
  corres10_idx1 = np.arange(len(nns10))
  corres10_idx0 = nns10

  mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
  corres_idx0 = corres01_idx0[mutual_filter]
  corres_idx1 = corres01_idx1[mutual_filter]

  return corres_idx0, corres_idx1

def find_correspondences_parallel(feats0, feats1, mutual_filter=True):
    pool = multiprocessing.Pool()  # Create a pool of worker processes
    
    # Divide the feature indices into chunks
    chunk_size = 1000  # Adjust the chunk size based on your data size and available resources
    chunks0 = [feats0[i:i + chunk_size] for i in range(0, len(feats0), chunk_size)]
    
    # Create a partial function to pass arguments to find_knn_cpu
    partial_find_knn_cpu = partial(find_knn_cpu, feat1=feats1, knn=1, return_distance=False)
    
    # Perform parallel computation of nearest neighbors
    results = pool.map(partial_find_knn_cpu, chunks0)
    
    # Concatenate the results from all chunks
    nn_inds = np.concatenate(results)
    corres01_idx0 = np.arange(len(nn_inds))
    corres01_idx1 = nn_inds
    
    if not mutual_filter:
        return corres01_idx0, corres01_idx1
    
    # Perform mutual filtering
    chunks1 = [feats1[i:i + chunk_size] for i in range(0, len(feats1), chunk_size)]
    partial_find_knn_cpu_rev = partial(find_knn_cpu, feat1=feats0, knn=1, return_distance=False)
    results_rev = pool.map(partial_find_knn_cpu_rev, chunks1)
    nn_inds_rev = np.concatenate(results_rev)
    corres10_idx1 = np.arange(len(nn_inds_rev))
    corres10_idx0 = nn_inds_rev

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    pool.close()  # Close the pool of worker processes
    pool.join()   # Wait for all processes to finish
    
    return corres_idx0, corres_idx1

def corres_visualize(A_corr,B_corr,A_pcd,B_pcd):
    # visualize the point clouds together with feature correspondences
    points = np.concatenate((A_corr.T,B_corr.T),axis=0)
    lines = []
    num_corrs = A_corr.shape[1]
    for i in range(num_corrs):
        lines.append([i,i+num_corrs])
    colors = [[0, 1, 0] for i in range(len(lines))] # lines are shown in green
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([A_pcd,B_pcd,line_set])