import os, torch, time, shutil, json,glob,sys,copy, argparse
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset
from torch import optim, nn
import open3d as o3d
import time
cwd = os.getcwd()
sys.path.append(cwd)
from datasets.indoor import IndoorDataset
from datasets.kitti import KITTIDataset
from datasets.modelnet import ModelNetHdf
from datasets.dataloader import get_dataloader
from models.architectures import KPFCNN
from lib.utils import load_obj, setup_seed,natural_key, load_config
from lib.benchmark_utils import ransac_pose_estimation, to_o3d_pcd, get_blue, get_yellow, to_tensor, mutual_selection
import shutil
setup_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_point_cloud(path):
    # Initialize an empty PointCloud for safety
    point_cloud = o3d.geometry.PointCloud()

    if path.endswith(".ply") or path.endswith(".pcd"):
        point_cloud = o3d.io.read_point_cloud(path)
    elif path.endswith(".csv"):
        point_cloud_data = np.genfromtxt(path, delimiter=',', skip_header=1, usecols=(1, 2, 3))
        point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
    elif path.endswith(".bin"):
        with open(path, "rb") as file:
            data = file.read()
        array = np.frombuffer(data, dtype=np.float32)
        num_points = array.shape[0] // 4
        points = np.reshape(array, (num_points, 4))[:, :3]  # Extract XYZ coordinates
        point_cloud.points = o3d.utility.Vector3dVector(points)
    elif path.endswith(".off"):
        with open(path, 'r') as f:
            lines = f.readlines()
        # Check if the format is incorrect
        if "OFF" in lines[0] and not lines[0].startswith("OFF\n"):
            # Split the line after "OFF"
            header_line = lines[0].replace("OFF", "").strip()
            corrected_lines = ["OFF\n", header_line + "\n"] + lines[1:]

            # Write the corrected content back to the file
            with open(path, 'w') as f:
                f.writelines(corrected_lines)

        mesh = o3d.io.read_triangle_mesh(path)
        point_cloud = mesh.sample_points_uniformly(number_of_points=2000)       
    else:
        print("Unsupported file format!")
    return point_cloud

class ThreeDMatchDemo(Dataset):
    """
    Load subsampled coordinates, relative rotation and translation
    Output(torch.Tensor):
        src_pcd:        [N,3]
        tgt_pcd:        [M,3]
        rot:            [3,3]
        trans:          [3,1]
    """
    def __init__(self,config, src_path, tgt_path, VOXEL_SIZE = 0.025, transformation = None):
        super(ThreeDMatchDemo,self).__init__()
        self.config = config
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.VOXEL_SIZE = VOXEL_SIZE
        self.transformation = transformation

    def __len__(self):
        return 1

    def __getitem__(self,item): 
        # get pointcloud
        src_pcd = read_point_cloud(self.src_path)
        tgt_pcd = read_point_cloud(self.tgt_path)
        src_pcd = src_pcd.voxel_down_sample(self.VOXEL_SIZE)
        tgt_pcd = tgt_pcd.voxel_down_sample(self.VOXEL_SIZE)
        if self.transformation is not None:
            tgt_pcd = tgt_pcd.transform(self.transformation)
        src_pcd = np.array(src_pcd.points).astype(np.float32)
        tgt_pcd = np.array(tgt_pcd.points).astype(np.float32)
        tgt_pcd = add_gaussian_noise_to_subset(tgt_pcd)
        print(f"src_pcd size: {src_pcd.shape}")

        src_feats=np.ones_like(src_pcd[:,:1]).astype(np.float32)
        tgt_feats=np.ones_like(tgt_pcd[:,:1]).astype(np.float32)

        # fake the ground truth information
        rot = np.eye(3).astype(np.float32)
        trans = np.ones((3,1)).astype(np.float32)
        correspondences = torch.ones(1,2).long()

        return src_pcd,tgt_pcd,src_feats,tgt_feats,rot,trans, correspondences, src_pcd, tgt_pcd, torch.ones(1)

def main(config, demo_loader):
    config.model.eval()
    c_loader_iter = demo_loader.__iter__()
    with torch.no_grad():
        inputs = c_loader_iter.next()
        ##################################
        # load inputs to device.
        for k, v in inputs.items():  
            if type(v) == list:
                inputs[k] = [item.to(config.device) for item in v]
            else:
                inputs[k] = v.to(config.device)

        ###############################################
        # forward pass
        feats, scores_overlap, scores_saliency = config.model(inputs)  #[N1, C1], [N2, C2]
        pcd = inputs['points'][0]
        len_src = inputs['stack_lengths'][0][0]
        src_pcd, tgt_pcd = pcd[:len_src], pcd[len_src:]
        src_feats, tgt_feats = feats[:len_src].detach().cpu(), feats[len_src:].detach().cpu()
        src_overlap, src_saliency = scores_overlap[:len_src].detach().cpu(), scores_saliency[:len_src].detach().cpu()
        tgt_overlap, tgt_saliency = scores_overlap[len_src:].detach().cpu(), scores_saliency[len_src:].detach().cpu()


        ########################################
        # do probabilistic sampling guided by the score
        src_scores = src_overlap * src_saliency
        tgt_scores = tgt_overlap * tgt_saliency
        if(src_pcd.size(0) > inputs['stack_lengths'][0][0]):
            idx = np.arange(src_pcd.size(0))
            probs = (src_scores / src_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            src_pcd, src_feats = src_pcd[idx], src_feats[idx]
        if(tgt_pcd.size(0) > inputs['stack_lengths'][0][1]):
            idx = np.arange(tgt_pcd.size(0))
            probs = (tgt_scores / tgt_scores.sum()).numpy().flatten()
            idx = np.random.choice(idx, size= config.n_points, replace=False, p=probs)
            tgt_pcd, tgt_feats = tgt_pcd[idx], tgt_feats[idx]
        
        return src_feats, tgt_feats
def presetting_indoor(config_path, src_pcd_path, tgt_pcd_path, VOXEL_SIZE, custom_weights_path, transformation):
    
    args = {
        'config': config_path  
    }
    config = load_config(args['config'])
    config = edict(config)
    
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # model initialization
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')
    config.model = KPFCNN(config).to(config.device)

    # create dataset and dataloader
    demo_set = ThreeDMatchDemo(config, src_pcd_path, tgt_pcd_path, VOXEL_SIZE, transformation)

    demo_loader, _ = get_dataloader(dataset=demo_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1)

    # load your own weights
    state = torch.load(custom_weights_path)
    config.model.load_state_dict(state['state_dict'])
    return config, demo_loader
def presetting_outdoor(config_path, src_pcd_path, tgt_pcd_path, VOXEL_SIZE, custom_weights_path):
    
    args = {
        'config': config_path  
    }
    config = load_config(args['config'])
    config = edict(config)
    
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # model initialization
    config.architecture = [
        'simple',
        'resnetb',
    ]
    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
    
    config.architecture.append('nearest_upsample')
    config.architecture.append('last_unary')
    config.model = KPFCNN(config).to(config.device)

    # create dataset and dataloader
    demo_set = ThreeDMatchDemo(config, src_pcd_path, tgt_pcd_path, VOXEL_SIZE)


    demo_loader, _ = get_dataloader(dataset=demo_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1)

    # load your own weights
    state = torch.load(custom_weights_path)
    config.model.load_state_dict(state['state_dict'])
    return config, demo_loader
def presetting_modelnet(config_path, src_pcd_path, tgt_pcd_path, VOXEL_SIZE, custom_weights_path, transformation):
    
    args = {
        'config': config_path  
    }
    config = load_config(args['config'])
    config = edict(config)
    
    if config.gpu_mode:
        config.device = torch.device('cuda')
    else:
        config.device = torch.device('cpu')

    # model initialization
    config.architecture = [
        'simple',
        'resnetb',
        'resnetb',
    ]

    for i in range(config.num_layers-1):
        config.architecture.append('resnetb_strided')
        config.architecture.append('resnetb')
        config.architecture.append('resnetb')
    for i in range(config.num_layers-2):
        config.architecture.append('nearest_upsample')
        config.architecture.append('unary')
        config.architecture.append('unary')
    
    config.architecture.append('nearest_upsample')
    config.architecture.append('unary')
    config.architecture.append('last_unary')
    config.model = KPFCNN(config).to(config.device)

     # create dataset and dataloader


    # create dataset and dataloader
    demo_set = ThreeDMatchDemo(config, src_pcd_path, tgt_pcd_path, VOXEL_SIZE, transformation)
    
    demo_loader, _ = get_dataloader(dataset=demo_set,
                                        batch_size=config.batch_size,
                                        shuffle=False,
                                        num_workers=1)

    # load your own weights
    state = torch.load(custom_weights_path)
    config.model.load_state_dict(state['state_dict'])
    return config, demo_loader
if __name__=="__main__":
  directory = "/path/to/dir/"  # Replace with the actual directory path
  file_extension = ".ply"
  depth_files = [file for file in os.listdir(directory) if file.endswith(file_extension)]
  depth_files_sorted = sorted(depth_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
  depth_files_full_path = [os.path.join(directory, file) for file in depth_files_sorted]
  weight_path = './weights/indoor.pth'
  config_path = './configs/test/indoor.yaml'
  VOXEL_SIZE = 0.1
  for j in range(len(depth_files_full_path)-1):
      start_time = time.time()
      # start Predator
      config, demo_loader = presetting_indoor(config_path, depth_files_full_path[j], depth_files_full_path[j+1], VOXEL_SIZE, weight_path)
      
      # do feature extraction
      src_feats, tgt_feats = main(config, demo_loader)
      src_feat, tgt_feat = to_tensor(src_feats), to_tensor(tgt_feats)
      scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0,1).to(device)).cpu()
      selection = mutual_selection(scores[None,:,:])[0]
      row_sel, col_sel = np.where(selection)
      end_time = time.time()
      runtime = end_time - start_time
      # save result
      savepath = '/path/to/save/'
      if not os.path.exists(savepath):
          os.makedirs(f'{savepath}/runtime/')
      if not os.path.exists(f'{savepath}correspondences/{j}{j+1}/'):
          os.makedirs(f'{savepath}correspondences/{j}{j+1}/')  
      np.savetxt(f'{savepath}runtime/runtime{j}{j+1}.txt', [runtime], fmt='%.4f')
      np.savetxt(f'{savepath}correspondences/{j}{j+1}/corres_idx0.txt', row_sel, fmt='%d')
      np.savetxt(f'{savepath}correspondences/{j}{j+1}/corres_idx1.txt', col_sel, fmt='%d')
