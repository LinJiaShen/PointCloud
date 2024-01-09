import os
import sys
import math
import logging
import open3d as o3d
import numpy as np
import time
import torch
import copy
import MinkowskiEngine as ME
import subprocess
import re
sys.path.append('.')
from util.knn import find_knn_gpu
from model import load_model



def preprocess(pcd, voxel_size):
    '''
    Stage 0: preprocess raw input point cloud
    Input: raw point cloud
    Output: voxelized point cloud with
    - xyz:    unique point cloud with one point per voxel
    - coords: coords after voxelization
    - feats:  dummy feature placeholder for general sparse convolution
    '''
    if isinstance(pcd, o3d.geometry.PointCloud):
      xyz = np.array(pcd.points)
    elif isinstance(pcd, np.ndarray):
      xyz = pcd
    else:
      raise Exception('Unrecognized pcd type')

    # Voxelization:
    # Maintain double type for xyz to improve numerical accuracy in quantization
    sel = ME.utils.sparse_quantize(xyz / voxel_size, return_index=True)
    npts = len(sel[0])

    xyz = torch.from_numpy(xyz[sel[1]])

    # ME standard batch coordinates
    coords = ME.utils.batched_coordinates([torch.floor(xyz / voxel_size).int()])
    feats = torch.ones(npts, 1)

    return xyz.float(), coords, feats


def fcgf_feature_extraction(feats, coords, fcgf_model, device):
    '''
    Step 1: extract fast and accurate FCGF feature per point
    '''
    sinput = ME.SparseTensor(feats, coordinates=coords, device = device)

    return fcgf_model(sinput).F

def fcgf_feature_matching(feats0, feats1, network_config):
    '''
    Step 2: coarsely match FCGF features to generate initial correspondences
    '''
    nns = find_knn_gpu(feats0,
                       feats1,
                       nn_max_n = network_config.nn_max_n,
                       knn=1,
                       return_distance=False)
    corres_idx0 = torch.arange(len(nns)).long().squeeze()
    corres_idx1 = nns.long().squeeze()

    return corres_idx0, corres_idx1

def main_extract(path0, path1, weights, VOXEL_SIZE):
    # preprocessing
    pcd0 = read_point_cloud(path0).voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd0.estimate_normals()
    pcd1 = read_point_cloud(path1).voxel_down_sample(voxel_size=VOXEL_SIZE)
    pcd1.estimate_normals()

    # torch and model load
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(weights, map_location= device)
    network_config = state['config']
    inlier_feature_type = network_config.inlier_feature_type
    print(inlier_feature_type)
    voxel_size = network_config.voxel_size
    #print(f'=> Setting voxel size to {voxel_size}')

    # FCGF network initialization
    num_feats = 1
    try:
        FCGFModel = load_model(network_config['feat_model'])
        fcgf_model = FCGFModel(
            num_feats,
            network_config['feat_model_n_out'],
            bn_momentum=network_config['bn_momentum'],
            conv1_kernel_size=network_config['feat_conv1_kernel_size'],
            normalize_feature=network_config['normalize_feature'])

    except KeyError:  # legacy pretrained models
        FCGFModel = load_model(network_config['model'])
        fcgf_model = FCGFModel(num_feats,
                                    network_config['model_n_out'],
                                    bn_momentum=network_config['bn_momentum'],
                                    conv1_kernel_size=network_config['conv1_kernel_size'],
                                    normalize_feature=network_config['normalize_feature'])

    fcgf_model.load_state_dict(state['state_dict'])
    fcgf_model = fcgf_model.to(device)
    fcgf_model.eval()
    # Inlier network initialization
    num_feats = 6 if network_config.inlier_feature_type == 'coords' else 1
    InlierModel = load_model(network_config['inlier_model'])
    inlier_model = InlierModel(
        num_feats,
        1,
        bn_momentum=network_config['bn_momentum'],
        conv1_kernel_size=network_config['inlier_conv1_kernel_size'],
        normalize_feature=False,
        D=6)

    inlier_model.load_state_dict(state['state_dict_inlier'])
    inlier_model = inlier_model.to(device)
    inlier_model.eval()
    #print("=> loading finished")


    # Step 0: voxelize and generate sparse input
    xyz0, coords0, feats0 = preprocess(pcd0, voxel_size)
    xyz1, coords1, feats1 = preprocess(pcd1, voxel_size)
    start = time.time()
    # Step 1: Feature extraction
    fcgf_feats0 = fcgf_feature_extraction(feats0, coords0, fcgf_model, device)
    fcgf_feats1 = fcgf_feature_extraction(feats1, coords1, fcgf_model, device)
    # Step 2: Coarse correspondences
    corres_idx0, corres_idx1 = fcgf_feature_matching(fcgf_feats0, fcgf_feats1, network_config)
    finish = time.time()
    elapsed_time = finish - start

    corres_xyz0 = xyz0[corres_idx0].numpy()
    corres_xyz1 = xyz1[corres_idx1].numpy()
    return corres_idx0, corres_idx1, corres_xyz0, corres_xyz1, elapsed_time

if __name__ == "__main__":
  directory = "/path/to/dir/"  # Replace with the actual directory path
  file_extension = ".ply"
  model = "ResUNetBN2C-feat32-3dmatch-v0.05.pth" # default indoor model
  depth_files = [file for file in os.listdir(directory) if file.endswith(file_extension)]
  depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
  depth_files_full_path = [os.path.join(directory, file) for file in depth_files_sorted]
  
  for i in range(len(depth_files_full_path)-1):
      corres_xyz0, corres_xyz1, elapsed_time = main_extract(depth_files_full_path[i], depth_files_full_path[i+1], model, 0.1)
      corr_output_file = f'/path/to/correspondence/correspondences{i}{i+1}.txt'
      with open(corr_output_file, 'w') as file:
          for j in range(len(corres_xyz0)):
              line = f"{corres_xyz0[j][0]} {corres_xyz0[j][1]} {corres_xyz0[j][2]} {corres_xyz1[j][0]} {corres_xyz1[j][1]} {corres_xyz1[j][2]}\n"
              file.write(line)
      runtime_output_file = f'/path/to/time/runtime{i}{i+1}.txt'
      with open(runtime_output_file, 'w') as file:
          file.write(f"{elapsed_time:.2f}")


