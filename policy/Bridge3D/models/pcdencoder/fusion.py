import os
import time
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
from PIL import Image
import plotly.graph_objects as go
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import mcubes
import trimesh
from PIL import Image
import open3d as o3d


def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
        depth:          [rfn,pn,1]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=pts.dtype)],1)
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=pts.dtype)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0]
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth
    return pts_2d, ~(invalid_mask[...,0]), depth

def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return: feats_inter: b,n,f
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)    # [srn,1,n,2]
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n
    feats_inter = feats_inter.permute(0,2,1)
    return feats_inter

def create_init_grid(boundaries, step_size):
    x_lower, x_upper = boundaries['x_lower'], boundaries['x_upper']
    y_lower, y_upper = boundaries['y_lower'], boundaries['y_upper']
    z_lower, z_upper = boundaries['z_lower'], boundaries['z_upper']
    x = torch.arange(x_lower, x_upper, step_size, dtype=torch.float32) + step_size / 2
    y = torch.arange(y_lower, y_upper, step_size, dtype=torch.float32) + step_size / 2
    z = torch.arange(z_lower, z_upper, step_size, dtype=torch.float32) + step_size / 2
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return coords, xx.shape

def instance2onehot(instance, N = None):
    # :param instance: [**dim] numpy array uint8, val from 0 to N-1
    # :return: [**dim, N] numpy array bool
    if N is None:
        N = instance.max() + 1
    if type(instance) is np.ndarray:
        assert instance.dtype == np.uint8
        out = np.zeros(instance.shape + (N,), dtype=bool)
        for i in range(N):
            out[..., i] = (instance == i)
    elif type(instance) is torch.Tensor:
        assert instance.dtype == torch.uint8
        # assert instance.min() == 0
        out = torch.zeros(instance.shape + (N,), dtype=torch.bool, device=instance.device)
        for i in range(N):
            out[..., i] = (instance == i)
    return out

def onehot2instance(one_hot_mask):
    # :param one_hot_mask: [**dim, N] numpy array float32 or bool (probalistic or not)
    # :return: [**dim] numpy array uint8, val from 0 to N-1
    if type(one_hot_mask) == np.ndarray:
        return np.argmax(one_hot_mask, axis=-1).astype(np.uint8)
    elif type(one_hot_mask) == torch.Tensor:
        return torch.argmax(one_hot_mask, dim=-1).to(dtype=torch.uint8)
    else:
        raise NotImplementedError

def _init_low_level_memory(lower_bound, higher_bound, voxel_size, voxel_num):
    def pcd_to_voxel(pcds):
        if type(pcds) == list:
            pcds = np.array(pcds)
        # The pc is in numpy array with shape (..., 3)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.floor((pcds - lower_bound) / voxel_size).astype(np.int32)
        return voxels

    def voxel_to_pcd(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The pc is in numpy array with shape (..., 3)
        pcds = voxels * voxel_size + lower_bound
        return pcds

    def voxel_to_index(voxels):
        if type(voxels) == list:
            voxels = np.array(voxels)
        # The voxel is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        indexes = (
            voxels[..., 0] * voxel_num[1] * voxel_num[2]
            + voxels[..., 1] * voxel_num[2]
            + voxels[..., 2]
        )
        return indexes

    def index_to_voxel(indexes):
        if type(indexes) == list:
            indexes = np.array(indexes)
        # The index is in numpy array with shape (...,)
        # The voxel is in numpy array with shape (..., 3)
        voxels = np.zeros((indexes.shape + (3,)), dtype=np.int32)
        voxels[..., 2] = indexes % voxel_num[2]
        indexes = indexes // voxel_num[2]
        voxels[..., 1] = indexes % voxel_num[1]
        voxels[..., 0] = indexes // voxel_num[1]
        return voxels

    def pcd_to_index(pcds):
        # The pc is in numpy array with shape (..., 3)
        # The index is in numpy array with shape (...,)
        voxels = pcd_to_voxel(pcds)
        indexes = voxel_to_index(voxels)
        return indexes

    def index_to_pcd(indexes):
        # The index is in numpy array with shape (...,)
        # The pc is in numpy array with shape (..., 3)
        voxels = index_to_voxel(indexes)
        pcds = voxel_to_pcd(voxels)
        return pcds

    return (
        pcd_to_voxel,
        voxel_to_pcd,
        voxel_to_index,
        index_to_voxel,
        pcd_to_index,
        index_to_pcd,
    )

class Fusion():
    def __init__(self, num_cam, feat_backbone='dinov3', device='cuda:0', dtype=torch.float32,
                 build_now=False, dinov3_weights=None, dinov3_repo=None):
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        self.dinov3_weights = dinov3_weights
        self.dinov3_repo = dinov3_repo
        
        # hyper-parameters
        self.mu = 0.02
        
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam
        self.feat_backbone = feat_backbone
        
        self._built = False      # NEW FLAG
        if build_now:
            self._build_external_models()
        
    def _build_external_models(self):
        if self._built:
            return
        if self.feat_backbone != 'dinov3':
            raise NotImplementedError(f"Unsupported point-cloud backbone: {self.feat_backbone}")
        if torch.__version__ < (2, 2):
            raise RuntimeError(
                f"The official DINOv3 source requires PyTorch >= 2.2; found {torch.__version__}. "
                "Install a matching torchvision version and rebuild PyTorch-dependent CUDA "
                "extensions before using this backbone. See models/pcdencoder/README.md."
            )

        project_root = Path(__file__).resolve().parents[2]
        weights = self.dinov3_weights or os.environ.get('DINOV3_WEIGHTS')
        if not weights:
            weights = project_root.parent / 'weights/DINOv3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
        weights = str(weights)
        if not weights.startswith('https://'):
            weights_path = Path(weights).expanduser()
            if not weights_path.is_absolute():
                weights_path = project_root / weights_path
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"DINOv3 ViT-L/16 weights not found: {weights_path}. "
                    "Download the official LVD-1689M checkpoint and set DINOV3_WEIGHTS "
                    "to its path or download URL."
                )
            weights = str(weights_path.resolve())

        repo_dir = self.dinov3_repo or os.environ.get('DINOV3_REPO')
        if repo_dir:
            repo_path = Path(repo_dir).expanduser()
            if not repo_path.is_absolute():
                repo_path = project_root / repo_path
            if not (repo_path / 'hubconf.py').is_file():
                raise FileNotFoundError(f"DINOv3 repository with hubconf.py not found: {repo_path}")
            repo, source = str(repo_path.resolve()), 'local'
        else:
            repo, source = 'facebookresearch/dinov3', 'github'

        model = torch.hub.load(
            repo,
            'dinov3_vitl16',
            source=source,
            pretrained=True,
            weights=weights,
            trust_repo=True,
        )
        self.patch_size = int(model.patch_size)
        self.feat_dim = int(model.embed_dim)
        if self.patch_size != 16 or self.feat_dim != 1024:
            raise ValueError(
                f"Expected DINOv3 ViT-L/16 with patch size 16 and 1024 features, "
                f"got patch size {self.patch_size} and {self.feat_dim} features."
            )
        self.dinov3_feat_extractor = model.to(device=self.device, dtype=self.dtype).eval()
        self.dinov3_feat_extractor.requires_grad_(False)
        
        self._built = True
        
    def eval(self, pts, return_names=['dino_feats', 'mask'], return_inter=False):
        # :param pts: (N, 3) torch tensor in world frame
        # :param return_names: a set of {'dino_feats', 'mask'}
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        #          - 'dino_feats': (N, f) torch tensor, the features of the points
        #          - 'mask': (N, NQ) torch tensor, the query masks of the points
        #          - 'valid_mask': (N) torch tensor, whether the point is valid
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]

        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > 0.0) & valid_mask & (dist > -self.mu) # [rfn,pn]
        
        # distance-based weight
        dist_weight = torch.exp(torch.clamp(self.mu-torch.abs(dist), max=0) / self.mu) # [rfn,pn]
        
        
        dist = torch.clamp(dist, min=-self.mu, max=self.mu) # [rfn,pn]
        
        # # weighted distance
        # dist = (dist * dist_weight).sum(0) / (dist_weight.sum(0) + 1e-6) # [pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        dist[dist_all_invalid] = 1e3
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        for k in return_names:
            inter_k = interpolate_feats(self.curr_obs_torch[k].permute(0,3,1,2),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='bilinear') # [rfn,pn,k_dim]
            
            # weighted sum
            # val = (inter_k * dist_weight.unsqueeze(-1)).sum(0) / (dist_weight.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            
            # # valid-weighted sum
            val = (inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            val[dist_all_invalid] = 0.0
            
            outputs[k] = val
            if return_inter:
                outputs[k+'_inter'] = inter_k
            else:
                del inter_k
        
        return outputs

    def eval_dist(self, pts):
        # this version does not clamp the distance or change the invalid points to 1e3
        # this is for grasper planner to find the grasping pose that does not penalize the depth
        # :param pts: (N, 3) torch tensor in world frame
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]
        
        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > 0.0) & valid_mask # [rfn,pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        return outputs
        

    
    def batch_eval(self, pts, return_names=['dino_feats', 'mask']):
        batch_pts = 60000
        outputs = {}
        for i in range(0, pts.shape[0], batch_pts):
            st_idx = i
            ed_idx = min(i + batch_pts, pts.shape[0])
            out = self.eval(pts[st_idx:ed_idx], return_names=return_names)
            for k in out:
                if k not in outputs:
                    outputs[k] = [out[k]]
                else:
                    outputs[k].append(out[k])
        
        # concat the outputs
        for k in outputs:
            if outputs[k][0] is not None:
                outputs[k] = torch.cat(outputs[k], dim=0)
            else:
                outputs[k] = None
        return outputs
    
    
    def extract_dinov3_features(self, imgs, params):
        if not self._built:
            self._build_external_models()
        K = imgs.shape[0]
        patch_h = int(params['patch_h'])
        patch_w = int(params['patch_w'])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError("DINOv3 patch grid dimensions must be positive.")
        input_size = (patch_h * self.patch_size, patch_w * self.patch_size)

        transform = T.Compose([
            T.Resize(input_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        imgs_tensor = torch.stack([
            transform(Image.fromarray(img).convert('RGB')) for img in imgs
        ]).to(device=self.device, dtype=self.dtype)
        with torch.autocast(device_type=self.device.type, enabled=False), torch.no_grad():
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            start_time = time.perf_counter()
            features_dict = self.dinov3_feat_extractor.forward_features(imgs_tensor)
            if self.device.type == 'cuda':
                torch.cuda.synchronize(self.device)
            print(f"dinov3 time: {(time.perf_counter() - start_time) * 1000:.3f} ms")
            features = features_dict['x_norm_patchtokens']
            expected_shape = (K, patch_h * patch_w, self.feat_dim)
            if tuple(features.shape) != expected_shape:
                raise RuntimeError(
                    f"Unexpected DINOv3 patch features: {tuple(features.shape)}; "
                    f"expected {expected_shape}."
                )
        return features.reshape(K, patch_h, patch_w, self.feat_dim)

    def extract_features(self, imgs, params):
        # :param imgs (K, H, W, 3) np array, color images
        # :param params: dict contains:
        #                - 'patch_h', 'patch_w': int, the size of the patch
        # :return features: (K, patch_h, patch_w, feat_dim) np array, features of the images
        if self.feat_backbone == 'dinov3':
            return self.extract_dinov3_features(imgs, params)
        else:
            raise NotImplementedError
    
    def update(self, obs):
        
        if not self._built:
            self._build_external_models()
        # :param obs: dict contains:
        #             - 'color': (K, H, W, 3) np array, color image
        #             - 'depth': (K, H, W) np array, depth image
        #             - 'pose': (K, 4, 4) np array, camera pose
        #             - 'K': (K, 3, 3) np array, camera intrinsics
        self.num_cam = obs['color'].shape[0]
        color = obs['color']
        params = {
            'patch_h': color.shape[1] // 10,
            'patch_w': color.shape[2] // 10,
        }
        features = self.extract_features(color, params)
        
        # self.curr_obs_torch = {
        #     'dino_feats': features,
        #     'color': color,
        #     'depth': torch.from_numpy(obs['depth']).to(self.device, dtype=torch.float32),
        #     'pose': torch.from_numpy(obs['pose']).to(self.device, dtype=torch.float32),
        #     'K': torch.from_numpy(obs['K']).to(self.device, dtype=torch.float32),
        # }
        self.curr_obs_torch['dino_feats'] = features
        self.curr_obs_torch['color'] = color
        self.curr_obs_torch['color_tensor'] = torch.from_numpy(color).to(self.device, dtype=self.dtype) / 255.0
        self.curr_obs_torch['depth'] = torch.from_numpy(obs['depth']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).to(self.device, dtype=self.dtype)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).to(self.device, dtype=self.dtype)
        
        _, self.H, self.W = obs['depth'].shape
    
    def voxel_downsample(self, pcd, voxel_size):
        # :param pcd: [N,3] numpy array
        # :param voxel_size: float
        # :return: [M,3] numpy array
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
        pcd_down = pcd.voxel_down_sample(voxel_size)
        return np.asarray(pcd_down.points)
    
    def pcd_iou(self, pcd_1, pcd_2, threshold):
        # :param pcd_1 [N,3] numpy array
        # :param pcd_2 [M,3] numpy array
        # voxel downsample
        # voxel_size = threshold
        # pcd_1 = self.voxel_downsample(pcd_1, voxel_size)
        # pcd_2 = self.voxel_downsample(pcd_2, voxel_size)
        dist = np.linalg.norm(pcd_1[:,None] - pcd_2[None], axis=-1) # [N,M]
        min_dist_from_1_to_2 = dist.min(axis=1) # [N]
        min_idx_from_1_to_2 = dist.argmin(axis=1) # [N]
        min_dist_from_2_to_1 = dist.min(axis=0) # [M]
        min_idx_from_2_to_1 = dist.argmin(axis=0) # [M]
        iou = ((min_dist_from_1_to_2 < threshold).sum() + (min_dist_from_2_to_1 < threshold).sum()) / (pcd_1.shape[0] + pcd_2.shape[0])
        iou_1 = (min_dist_from_1_to_2 < threshold).sum() / pcd_1.shape[0]
        iou_2 = (min_dist_from_2_to_1 < threshold).sum() / pcd_2.shape[0]
        overlap_idx_1 = np.where(min_dist_from_1_to_2 < threshold)[0]
        overlap_idx_2 = np.where(min_dist_from_2_to_1 < threshold)[0]
        return iou, iou_1, iou_2, overlap_idx_1, overlap_idx_2, min_idx_from_1_to_2, min_idx_from_2_to_1
    
    def vox_idx_iou(self, vox_idx_1, vox_idx_2):
        intersection = len(
            set(vox_idx_1).intersection(set(vox_idx_2))
        )
        union = len(set(vox_idx_1).union(set(vox_idx_2)))
        return intersection / union, len(vox_idx_1) / union, len(vox_idx_2) / union
    
    def del_partial_pcd(self, instance_info, pcd_idx):
        start_idx = 0
        for view_idx in instance_info['pcd'].keys():
            end_idx = start_idx + instance_info['pcd'][view_idx].shape[0]
            pcd_idx_in_view = pcd_idx[np.logical_and(pcd_idx >= start_idx, pcd_idx < end_idx)]
            pcd_idx_in_view -= start_idx
            instance_info['pcd'][view_idx] = np.delete(instance_info['pcd'][view_idx], pcd_idx_in_view, axis=0)
            start_idx = end_idx
        return instance_info

    def del_partial_vox_idx(self, instance_info, vox_idx):
        curr_vox_idx = set(instance_info['vox_idx'])
        for vox_i in vox_idx:
            if vox_i in instance_info['conf_per_pt']:
                del instance_info['conf_per_pt'][vox_i]
            if vox_i in instance_info['vox_idx']:
                curr_vox_idx.remove(vox_i)
        instance_info['vox_idx'] = np.array(list(curr_vox_idx))
        return instance_info
    
    def filter_instances(self, instances_info):
        to_delete = []

        # Filter 1: filter instances that have a large IoU with other instances
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            for idx_j, instance_j in enumerate(instances_info):
                if idx_j <= idx_i:
                    continue
                if idx_j in to_delete:
                    continue
                
                pcd_i = np.concatenate([instance_i['pcd'][view_idx] for view_idx in instance_i['pcd'].keys()], axis=0)
                conf_per_pt_i = np.concatenate([np.ones(instance_i['pcd'][view_idx].shape[0]) * instance_i['conf'][view_idx] for view_idx in instance_i['pcd'].keys()])
                pcd_j = np.concatenate([instance_j['pcd'][view_idx] for view_idx in instance_j['pcd'].keys()], axis=0)
                conf_per_pt_j = np.concatenate([np.ones(instance_j['pcd'][view_idx].shape[0]) * instance_j['conf'][view_idx] for view_idx in instance_j['pcd'].keys()])
                iou, iou_1, iou_2, overlap_idx_1, overlap_idx_2, min_idx_from_1_to_2, min_idx_from_2_to_1 = self.pcd_iou(pcd_i, pcd_j, threshold=0.005)
                if iou > 0.25:
                    # we can only keep one
                    num_vis_view_i = len(instance_i['idx'])
                    num_vis_view_j = len(instance_j['idx'])
                    if num_vis_view_i > num_vis_view_j:
                        to_delete.append(idx_j)
                    elif num_vis_view_j > num_vis_view_i:
                        to_delete.append(idx_i)
                    else:
                        # # ver 1: delete the whole instance
                        # conf_i = np.mean([instance_i['conf'][view_idx] for view_idx in instance_i['conf'].keys()])
                        # conf_j = np.mean([instance_j['conf'][view_idx] for view_idx in instance_j['conf'].keys()])
                        # if conf_i > conf_j:
                        #     to_delete.append(idx_j)
                        # else:
                        #     to_delete.append(idx_i)
                        
                        # ver 2: delete the point with less confidence
                        overlap_conf_1 = conf_per_pt_i[overlap_idx_1]
                        overlap_conf_2_corr_to_1 = conf_per_pt_j[min_idx_from_1_to_2[overlap_idx_1]]
                        overlap_conf_2 = conf_per_pt_j[overlap_idx_2]
                        overlap_conf_1_corr_to_2 = conf_per_pt_i[min_idx_from_2_to_1[overlap_idx_2]]
                        pcd_i_del_idx = overlap_idx_1[overlap_conf_1 < overlap_conf_2_corr_to_1]
                        pcd_j_del_idx = overlap_idx_2[overlap_conf_2 < overlap_conf_1_corr_to_2]
                        
                        # remove points in instance_i
                        instance_i = self.del_partial_pcd(instance_i, pcd_i_del_idx)
                        instance_j = self.del_partial_pcd(instance_j, pcd_j_del_idx)
                
                # instance_i is a subset of instance_j
                elif iou_1 > 0.5:
                    # we may delete instance_i or remove overlapping points in instance_j
                    num_vis_view_i = len(instance_i['idx'])
                    num_vis_view_j = len(instance_j['idx'])
                    if (instance_j['label'] == 'background' and num_vis_view_i < self.num_cam // 2) or \
                        (instance_j['label'] != 'background' and num_vis_view_i < num_vis_view_j // 2):
                        # delete instance_i
                        to_delete.append(idx_i)
                    else:
                        # remove overlapping points in instance_j
                        instance_j = self.del_partial_pcd(instance_j, overlap_idx_2)
                
                # instance_j is a subset of instance_i
                elif iou_2 > 0.5:
                    # we may delete instance_j or remove overlapping points in instance_i
                    num_vis_view_i = len(instance_i['idx'])
                    num_vis_view_j = len(instance_j['idx'])
                    if (instance_i['label'] == 'background' and num_vis_view_j < self.num_cam // 2) or \
                        (instance_i['label'] != 'background' and num_vis_view_j < num_vis_view_i // 2):
                        # delete instance_j
                        to_delete.append(idx_j)
                    else:
                        # remove overlapping points in instance_i
                        instance_i = self.del_partial_pcd(instance_i, overlap_idx_1)
                
                # immediately put instance to delete if it is too small
                pcd_i = np.concatenate([instance_i['pcd'][view_idx] for view_idx in instance_i['pcd'].keys()], axis=0)
                if pcd_i.shape[0] < 10:
                    to_delete.append(idx_i)
                pcd_j = np.concatenate([instance_j['pcd'][view_idx] for view_idx in instance_j['pcd'].keys()], axis=0)
                if pcd_j.shape[0] < 10:
                    to_delete.append(idx_j)

        # Filter 2: filter certain instances used as background
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            bg_name_ls = ['table']
            if instance_i['label'] in bg_name_ls:
                to_delete.append(idx_i)
                continue
        
        # Filter 3: filter instances that are too small
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            pcd_i = np.concatenate([instance_i['pcd'][view_idx] for view_idx in instance_i['pcd'].keys()], axis=0)
            if pcd_i.shape[0] < 10:
                to_delete.append(idx_i)
                continue
        
        for idx in sorted(to_delete, reverse=True):
            del instances_info[idx]
        
        return instances_info
    
    def filter_instances_vox_ver(self, instances_info):
        to_delete = []

        # Filter 1: filter instances that have a large IoU with other instances
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            for idx_j, instance_j in enumerate(instances_info):
                if idx_j <= idx_i:
                    continue
                if idx_j in to_delete:
                    continue
                
                vox_idx_i = instance_i['vox_idx']
                conf_per_pt_i = instance_i['conf_per_pt']
                vox_idx_j = instance_j['vox_idx']
                conf_per_pt_j = instance_j['conf_per_pt']
                iou, iou_1, iou_2 = self.vox_idx_iou(vox_idx_i, vox_idx_j)
                if iou > 0.25 or iou_1 > 0.5 or iou_2 > 0.5:
                    # ver 2: delete the point with less views and less confidence
                    to_delete_i = []
                    for vox_i in conf_per_pt_i.keys():
                        # keep points that are not in vox_idx_j
                        if vox_i not in conf_per_pt_j:
                            continue
                        if len(conf_per_pt_i[vox_i]) < len(conf_per_pt_j[vox_i]):
                            to_delete_i.append(vox_i)
                        elif len(conf_per_pt_i[vox_i]) == len(conf_per_pt_j[vox_i]):
                            if np.mean(conf_per_pt_i[vox_i]) < np.mean(conf_per_pt_j[vox_i]):
                                to_delete_i.append(vox_i)
                    to_delete_j = []
                    for vox_j in conf_per_pt_j.keys():
                        # keep points that are not in vox_idx_i
                        if vox_j not in conf_per_pt_i:
                            continue
                        if len(conf_per_pt_j[vox_j]) < len(conf_per_pt_i[vox_j]):
                            to_delete_j.append(vox_j)
                        elif len(conf_per_pt_j[vox_j]) == len(conf_per_pt_i[vox_j]):
                            if np.mean(conf_per_pt_j[vox_j]) < np.mean(conf_per_pt_i[vox_j]):
                                to_delete_j.append(vox_j)
                    
                    # remove points in instance_i
                    instance_i = self.del_partial_vox_idx(instance_i, to_delete_i)
                    instance_j = self.del_partial_vox_idx(instance_j, to_delete_j)
                
                # immediately put instance to delete if it is too small
                if len(instance_i['vox_idx']) < 1:
                    to_delete.append(idx_i)
                if len(instance_j['vox_idx']) < 1:
                    to_delete.append(idx_j)

        # Filter 2: filter certain instances used as background
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            bg_name_ls = ['table']
            if instance_i['label'] in bg_name_ls:
                to_delete.append(idx_i)
                continue
        
        # Filter 3: filter instances that are too small
        for idx_i, instance_i in enumerate(instances_info):
            if idx_i in to_delete:
                continue
            if len(instance_i['vox_idx']) < 1:
                to_delete.append(idx_i)
                continue
        
        for idx in sorted(to_delete, reverse=True):
            del instances_info[idx]
        
        return instances_info
    
    def reorder_instances(self, instances_info, query_texts):
        # reorder instances based on the query texts
        new_instances_info = []
        for query_text in ['background'] + query_texts:
            for instance_info in instances_info:
                if instance_info['label'] == query_text:
                    new_instances_info.append(instance_info)
        return new_instances_info

    def swap_instance_mask(self, instances_info):
        self.curr_obs_torch['mask'] = torch.zeros((self.num_cam, self.H, self.W), device=self.device, dtype=torch.uint8)
        for i in range(self.num_cam):
            new_mask = np.zeros_like(self.curr_obs_torch['mask_gs'][i][0], dtype=np.uint8)
            for inst_idx, instance_info in enumerate(instances_info):
                if i not in instance_info['idx']:
                    continue
                mask_idx = instance_info['idx'][i]
                new_mask[self.curr_obs_torch['mask_gs'][i][mask_idx]] = inst_idx
            self.curr_obs_torch['mask'][i] = torch.from_numpy(new_mask).to(device=self.device, dtype=torch.uint8)
    
            # self.curr_obs_torch[f'mask'] = query_mask # [num_cam, H, W]
    
    def get_inst_num(self):
        # NOTE: including the background
        return len(self.curr_obs_torch['consensus_mask_label'])
    
    def extract_mesh(self, pts, res, grid_shape):
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        # :param grid_shape: (3) tuple, the shape of the grid
        dist = res['dist'].detach().cpu().numpy()
        dist = dist.reshape(grid_shape)
        smoothed_dist = mcubes.smooth(dist)
        vertices, triangles = mcubes.marching_cubes(smoothed_dist, 0)
        vertices = vertices.astype(np.int32)
        # vertices_flat = np.unravel_index(vertices, grid_shape)
        vertices_flat = np.ravel_multi_index(vertices.T, grid_shape)
        vertices_coords = pts.detach().cpu().numpy()[vertices_flat]
        
        return vertices_coords, triangles
    
    def create_mask_mesh(self, vertices, triangles, res):
        # :param vertices: (N, 3) numpy array, the vertices of the mesh
        # :param triangles: (M, 3) numpy array, the triangles of the mesh
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        query_masks = res['query_masks'].detach().cpu().numpy()
        mask_meshes = []
        for i in range(query_masks.shape[1]):
            vertices_color = trimesh.visual.interpolate(query_masks[:,i], color_map='viridis')
            mask_meshes.append(trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=vertices_color))
        return mask_meshes

    def create_instance_mask_mesh(self, vertices, triangles, res):
        # :param vertices: (N, 3) numpy array, the vertices of the mesh
        # :param triangles: (M, 3) numpy array, the triangles of the mesh
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'mask_*': (N, nq) torch tensor, the instance masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        mask_meshes = []
        for k in res.keys():
            if k.startswith('mask'):
                mask = res[k].detach().cpu().numpy()
                num_instance = mask.shape[1]
                mask = onehot2instance(mask) # (N, nq) -> (N)
                
                
                vertices_color = trimesh.visual.interpolate(mask / num_instance, color_map='jet')
                mask_meshes.append(trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=vertices_color))
        return mask_meshes

    
    def create_descriptor_mesh(self, vertices, triangles, res, params, mask_out_bg):
        pca = params['pca']
        features = res['dino_feats'].detach().cpu().numpy()
        mask = res['mask'].detach().cpu().numpy()
        mask = onehot2instance(mask) # (N, nq) -> (N)
        bg = (mask == 0)
        features_pca = pca.transform(features)
        
        features_rgb = np.zeros((features_pca.shape[0], 3))
        
        for i in range(features_pca.shape[1]):
            if not mask_out_bg:
                features_rgb[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
            else:
                features_rgb[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
                # features_rgb[~bg, i] = (features_pca[~bg, i] - features_pca[~bg, i].min()) / (features_pca[~bg, i].max() - features_pca[~bg, i].min())
                # features_rgb[bg, i] = 0.8
        features_rgb[bg] = np.ones(3) * 0.8
        features_rgb = features_rgb[..., ::-1]
            
        features_rgb = (features_rgb * 255).astype(np.uint8)
        features_rgb = np.concatenate([features_rgb, np.ones((features_rgb.shape[0], 1), dtype=np.uint8) * 255], axis=1)
        features_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=features_rgb)
        return features_mesh

    def create_color_mesh(self, vertices, triangles, res):
        colors = res['color_tensor'].detach().cpu().numpy()[..., ::-1]
        colors = (colors * 255).astype(np.uint8)
        colors = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.uint8) * 255], axis=1)
        color_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=colors)
        return color_mesh
    
    def rigid_tracking(self,
                       src_feat_info,
                       last_match_pts_list,
                       boundaries,
                       rand_ptcl_num):
        lr = 0.01
        iter_num = 100
        reg_w = 1.0
        dist_w = 100.0
        oob_w = 0.0

        src_feats = [src_feat_info[k]['src_feats'] for k in src_feat_info.keys()]
        src_feats = torch.cat(src_feats, dim=0) # [num_instance * rand_ptcl_num, feat_dim]
        
        num_instance = len(last_match_pts_list)
        last_match_pts_np = np.stack(last_match_pts_list, axis=0) # [num_instance, rand_ptcl_num, 3]
        assert last_match_pts_np.shape[:2] == (num_instance, rand_ptcl_num)
        
        last_match_pts_tensor = torch.from_numpy(last_match_pts_np).to(self.device, dtype=torch.float32)
        from pytorch3d.transforms import Transform3d
        from pytorch3d.transforms.so3 import so3_exp_map
        # T = torch.zeros((num_instance, 3, 4)).to(self.device, dtype=torch.float32)
        # T.requires_grad_()
        # T = SE3.Identity(1).to(self.device, dtype=torch.float32)
        t_params = torch.zeros(num_instance, 3, requires_grad=True)
        # r_params = torch.randn(num_instance, 3, 3, requires_grad=True)
        # r_params = torch.eye(3).repeat(num_instance, 1, 1).to(self.device, dtype=torch.float32).requires_grad_()
        log_r_params = torch.zeros(num_instance, 3, requires_grad=True)
        optimizer = torch.optim.Adam([t_params, log_r_params], lr=lr, betas=(0.9, 0.999))
        
        loss_list = []
        feat_loss_list = []
        dist_loss_list = []
        reg_loss_list = []
        oob_loss_list = []
        for iter_idx in range(iter_num):
            # last_match_pts_tensor_homo = torch.cat([last_match_pts_tensor, torch.ones((num_instance, rand_ptcl_num, 1)).to(self.device, dtype=torch.float32)], dim=-1).permute(0, 2, 1) # [num_instance, 4, rand_ptcl_num]
            # curr_match_pts = torch.bmm(T, last_match_pts_tensor_homo).permute(0, 2, 1) # [num_instance, rand_ptcl_num, 3]
            # curr_match_pts = curr_match_pts.reshape(-1, 3)
            r_params = so3_exp_map(log_r_params)
            T = Transform3d(device=self.device, dtype=torch.float32).rotate(r_params).translate(t_params)
            curr_match_pts = T.transform_points(last_match_pts_tensor).reshape(-1, 3)
            out = self.eval(curr_match_pts, return_names=['dino_feats'])
            curr_feats = out['dino_feats'] # [num_instance * rand_ptcl_num, feat_dim]
            curr_dist = out['dist'] * out['valid_mask'] # [num_instance * rand_ptcl_num]
            feat_loss = (torch.norm((curr_feats - src_feats), dim=-1) * out['valid_mask']).mean()
            dist_loss = dist_w * torch.clamp(curr_dist, min=0).mean()
            reg_loss = reg_w * (torch.norm(t_params) + torch.norm(log_r_params))
            oob_loss = oob_w * (torch.clamp(curr_match_pts[:, 0] - boundaries['x_upper'], min=0) +\
                    torch.clamp(boundaries['x_lower'] - curr_match_pts[:, 0], min=0) +\
                    torch.clamp(curr_match_pts[:, 1] - boundaries['y_upper'], min=0) +\
                    torch.clamp(boundaries['y_lower'] - curr_match_pts[:, 1], min=0) +\
                    torch.clamp(curr_match_pts[:, 2] - boundaries['z_upper'], min=0) +\
                    torch.clamp(boundaries['z_lower'] - curr_match_pts[:, 2], min=0)).mean()
            loss = feat_loss + dist_loss + reg_loss # + oob_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss for debug
            loss_list.append(loss.item())
            feat_loss_list.append(feat_loss.item())
            dist_loss_list.append(dist_loss.item())
            reg_loss_list.append(reg_loss.item())
            oob_loss_list.append(oob_loss.item())
            
        # plt.plot(loss_list, label='loss')
        # plt.plot(feat_loss_list, label='feat_loss')
        # plt.plot(dist_loss_list, label='dist_loss')
        # plt.plot(reg_loss_list, label='reg_loss')
        # plt.legend()
        # plt.show()
        
        match_pts_list = []
        for i in range(num_instance):
            match_pts_list.append(curr_match_pts[i * rand_ptcl_num:(i + 1) * rand_ptcl_num].detach().cpu().numpy())
        
        return {'match_pts_list': match_pts_list,}

    def vis_curr_mask(self):
        # return vis_mask, which is a numpy array of shape [num_cam, H, W, 3]. It can be visualized using cv2
        vis_mask = np.zeros((self.num_cam, self.H, self.W, 3))
        for i in range(self.num_cam):
            color = (self.curr_obs_torch['color'][i] * 255).astype(np.uint8)
            mask = self.curr_obs_torch['mask'][i].detach().cpu().numpy() # [H, W, num_instance]
            mask = onehot2instance(mask) # [H, W]
            jet_cmap = cm.get_cmap('jet')
            mask_colors = (jet_cmap(mask / mask.max())[..., :3] * 255).astype(np.uint8)
            vis_mask[i] = (0.5 * color + 0.5 * mask_colors).astype(np.uint8)
        return vis_mask.astype(np.uint8)
    
    def close(self):
        self.curr_obs_torch.clear()
        if hasattr(self, 'dinov3_feat_extractor'):
            del self.dinov3_feat_extractor
        self._built = False

def compare_dinov3_time():
    model_type = 'dinov3'
    device = 'cuda'
    fusion = Fusion(num_cam=4, feat_backbone=model_type, device=device, dtype=torch.float16)
    for i in range(11):
        if i == 1:
            start_time = time.time()
        img = (np.random.randn(4, 240, 320, 3) * 255).astype(np.uint8)
        params = {
            'patch_h': img.shape[1] // 10,
            'patch_w': img.shape[2] // 10,
        }
        fusion.extract_features(img, params)
    print(f'{model_type} time: {time.time() - start_time}')

def compare_float_prec():
    model_type = 'dinov3'
    device = 'cuda'
    fusion = Fusion(num_cam=4, feat_backbone=model_type, device=device, dtype=torch.float16)
    img = (np.random.randn(4, 240, 320, 3) * 255).astype(np.uint8)
    params = {
        'patch_h': img.shape[1] // 10,
        'patch_w': img.shape[2] // 10,
    }
    feats = fusion.extract_features(img, params)
    
    fusion_full = Fusion(num_cam=4, feat_backbone=model_type, device=device, dtype=torch.float32)
    feats_full = fusion_full.extract_features(img, params)
    
    print('feat diff: ', torch.norm(feats - feats_full, dim=-1).mean())
    print('feat diff max: ', torch.max(torch.abs(feats - feats_full)))
