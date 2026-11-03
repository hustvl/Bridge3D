import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
from sklearn.decomposition import PCA
import trimesh

from .pointnet2 import PointNet2DenseEncoder

sys.path.append(os.getcwd())
from .fusion import Fusion, create_init_grid
from .draw_utils import aggr_point_cloud_from_data
from pytorch3d.ops import sample_farthest_points
from torch.nn.utils.rnn import pad_sequence


def onehot2instance(one_hot_mask):
    if isinstance(one_hot_mask, np.ndarray):
        return np.argmax(one_hot_mask, axis=-1).astype(np.uint8)
    elif isinstance(one_hot_mask, torch.Tensor):
        return torch.argmax(one_hot_mask, dim=-1).to(dtype=torch.uint8)
    else:
        raise NotImplementedError


def robust_minmax(x: np.ndarray, lo: float = 2.0, hi: float = 98.0, eps: float = 1e-6) -> np.ndarray:
    lo_v = np.percentile(x, lo, axis=0)
    hi_v = np.percentile(x, hi, axis=0)
    den = np.maximum(hi_v - lo_v, eps)
    return np.clip((x - lo_v) / den, 0.0, 1.0)


@dataclass
class PcdEncoderCfg:       
    num_cam: int = 1      
    x_upper: float = 1.2
    x_lower: float = -1.2
    y_upper: float = 1.2
    y_lower: float = -1.2
    z_upper: float = 1.2
    z_lower: float = -1.2

    feat_backbone: str = "dinov3"
    device: str = "cuda"
    
    pointcloud_encoder_cfg_backgroud: Optional[Dict] = field(default_factory=dict)
    pointcloud_encoder_cfg_target: Optional[Dict] = field(default_factory=dict)
    dinov3_weights: Optional[str] = None
    dinov3_repo: Optional[str] = None


class PcdEncoder(nn.Module):

    def __init__(self, cfg: PcdEncoderCfg):
        super().__init__()   
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() and "cuda" in cfg.device else "cpu"
        self.boundaries = {
            "x_lower": cfg.x_lower, "x_upper": cfg.x_upper,
            "y_lower": cfg.y_lower, "y_upper": cfg.y_upper,
            "z_lower": cfg.z_lower, "z_upper": cfg.z_upper
        }
        self.fusion = Fusion(
            num_cam=cfg.num_cam,
            feat_backbone=cfg.feat_backbone,
            device=self.device,
            dinov3_weights=cfg.dinov3_weights,
            dinov3_repo=cfg.dinov3_repo,
        )

        self.colors: Optional[np.ndarray] = None     # [N, H, W, C]
        self.depths: Optional[np.ndarray] = None     # [N, H, W]
        self.extrinsics: Optional[np.ndarray] = None # [N, 4, 4] or [N, 3, 4]
        self.intrinsics: Optional[np.ndarray] = None # [N, 3, 3]
        self.obs: Optional[Dict] = None
        self.pcd: Optional[o3d.geometry.PointCloud] = None
        self.pcd_vis: Optional[o3d.geometry.PointCloud] = None
    
        
    def _is_tensor(self, data):
        for k, v in data.items():
            if not torch.is_tensor(v):
                data[k] = torch.from_numpy(v)
        return data

    # ---------------- I/O ----------------
    def _read_rgbd_and_calib(self, pcd_data, batch_idx: int = 0):
        n = self.cfg.num_cam

        pcd_data = self._is_tensor(pcd_data)
        colors = []
        depths = []
        # intrinsics = torch.zeros((n, 3, 3), dtype=torch.float64, device='self.device') 
        intrinsics = torch.zeros((n, 3, 3), dtype=torch.float64, device='cpu') 
        extrinsics = []
        masks = []

        for _ in range(n):
            
            color = pcd_data['color'][batch_idx]                             # BGR uint8  (240, 320 ,3)
            depth = pcd_data['depth'][batch_idx]     # torch.float64  (240, 320)
            ext = pcd_data['extrinsics'][:, 0][batch_idx]                              # (4,4) or (3,4) torch.float32   (4, 4)
            if ext.shape == (3,4):
                ext_4x4 = torch.eye(4, dtype=ext.dtype, device=ext.device)
                ext_4x4[:3, :4] = ext
                ext = ext_4x4
            cam_param = pcd_data['intrinsics']                               # fx, fy, cx, cy torch.float64  (2, 1, 4)
            mask = pcd_data['pcd_mask'][batch_idx]                       # bool torch.uint8 (240, 320, 2)

            colors.append(color)
            depths.append(depth)
            extrinsics.append(ext)
            masks.append(mask)
        
            
        intrinsics[:, 0, 0] = cam_param[0][:, 0]
        intrinsics[:, 1, 1] = cam_param[0][:, 1]
        intrinsics[:, 0, 2] = cam_param[0][:, 2]
        intrinsics[:, 1, 2] = cam_param[0][:, 3]
        intrinsics[:, 2, 2] = 1.0

        self.colors = torch.stack(colors, dim=0)        # torch.Size([1, 240, 320, 3])
        self.depths = torch.stack(depths, dim=0)        # torch.Size([1, 240, 320])
        self.extrinsics = torch.stack(extrinsics, dim=0)  # torch.Size([1, 4, 4])
        self.intrinsics = intrinsics  # torch.Size([1, 3, 3])
        self.masks = torch.stack(masks, dim=0)  # torch.Size([1, 240, 320, 2])

    def _build_obs(self):
        assert self.colors is not None and self.depths is not None
        pose_3x4 = self.extrinsics[:, :3] if self.extrinsics.shape[-2:] == (4, 4) else self.extrinsics
        
        color_np = self.colors.detach().to('cpu').numpy()         # [N,H,W,3] BGR
        depth_np = self.depths.detach().to('cpu').numpy()         # [N,H,W]
        pose_np  = pose_3x4.to('cpu').numpy()     # [N,3,4]
        K_np     = self.intrinsics.detach().to('cpu').numpy()     # [N,3,3]
        
        self.obs = {
            "color": color_np,          # [N,H,W,3] BGR
            "depth": depth_np,          # [N,H,W]
            "pose": pose_np,              # (N,3,4)
            "K": K_np,          # (N,3,3)
        }

    # ---------------- Pipeline steps ----------------
    def build_point_cloud(self, downsample: bool = True, nb_neighbors: int = 20, std_ratio: float = 0.2):
        assert self.obs is not None
        pcd = aggr_point_cloud_from_data(
            self.obs["color"][..., ::-1],  # to RGB
            self.obs["depth"],
            self.obs["K"],
            self.extrinsics.detach().to('cpu').numpy(),
            downsample=downsample,
            boundaries=self.boundaries
        )
        pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        self.pcd = pcd

    def run_fusion(self):
        assert self.obs is not None
        self.fusion.update(self.obs)
        self.fusion.curr_obs_torch['mask'] = self.masks.to(self.fusion.device, dtype=self.fusion.dtype)

    def eval_vertices(self):
        assert self.pcd is not None
        pts = np.asarray(self.pcd.points, dtype=np.float32)
        vertices_tensor = torch.from_numpy(pts).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            out = self.fusion.batch_eval(vertices_tensor, return_names=["dino_feats", "mask", "color_tensor"])
        return pts, out

    def pca_colorize(self, feats: torch.Tensor) -> np.ndarray:
        # feats: (N,C) tensor
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        feat_np = feats.detach().cpu().numpy()
        pca = PCA(n_components=3, random_state=0)
        feat3 = pca.fit_transform(feat_np)          # (N,3)
        feat3_01 = robust_minmax(feat3).astype(np.float64)
        return feat3_01

            
    @staticmethod
    def to_mask_colors(points: np.ndarray, out_mask: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> o3d.geometry.PointCloud:
        labels = torch.argmax(out_mask, dim=1).cpu().numpy()  # (N,)
        N = points.shape[0]

        rng = np.random.default_rng(0)
        palette = {0: np.array([0.7, 0.7, 0.7]),
                   1: np.array([1.0, 0.2, 0.2])}
        num_classes = out_mask.shape[1]
        for cid in range(2, num_classes):
            if cid not in palette:
                palette[cid] = rng.random(3) * 0.8 + 0.2

        colors = np.zeros((N, 3), dtype=float)
        for cid in range(num_classes):
            colors[labels == cid] = palette[cid]

        if valid_mask is not None:
            valid = valid_mask.cpu().numpy().astype(bool)
            colors[~valid] = np.array([0.2, 0.2, 0.2])

        pcd_col = o3d.geometry.PointCloud()
        pcd_col.points = o3d.utility.Vector3dVector(points)
        pcd_col.colors = o3d.utility.Vector3dVector(colors)
        return pcd_col
    
    @staticmethod
    def pca_colors(points: np.ndarray, feats: torch.Tensor) -> o3d.geometry.PointCloud:
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        feat_np = feats.detach().cpu().numpy()
        pca = PCA(n_components=3, random_state=0)
        feat3 = pca.fit_transform(feat_np)
        feat3_01 = robust_minmax(feat3).astype(np.float64)

        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(points)
        pcd_vis.colors = o3d.utility.Vector3dVector(feat3_01)
        return pcd_vis

    def _module_device(self, module):
        return next(module.parameters()).device
    
    def pcd_background_fps(self, pcd, pcd_dino, pcd_rgb):
        
        pcd = pcd.unsqueeze(0)
        pcd_dino = pcd_dino.unsqueeze(0)
        pcd_rgb = pcd_rgb.unsqueeze(0)    # [1, N, 3]
        torch.cuda.empty_cache()
        # sampled_pcd_coord, idx = sample_farthest_points(pcd, K=256)
        sampled_pcd_coord, idx = sample_farthest_points(pcd, K=2)
        torch.cuda.empty_cache()
        B, K = idx.shape
        batch = torch.arange(B, device=pcd_dino.device).unsqueeze(1).expand(B, K)  # [B,K]
        sampled_pcd_feat = pcd_dino[batch, idx, :]   # -> [B, K, C]
        sampled_pcd_rgb  = pcd_rgb[batch, idx, :]    # [1, K, 3]
        
        return sampled_pcd_coord, sampled_pcd_feat, sampled_pcd_rgb
    
    def pcd_target_fps(self, pcd, pcd_dino, pcd_rgb):
        
        pcd = pcd.unsqueeze(0)
        pcd_dino = pcd_dino.unsqueeze(0)
        pcd_rgb = pcd_rgb.unsqueeze(0)    # [1, N, 3]
        torch.cuda.empty_cache()
        # K = min(pcd.shape[1], 1536)
        K = min(pcd.shape[1], 128)
        sampled_pcd_coord, idx = sample_farthest_points(pcd, K=K)
        torch.cuda.empty_cache()
        B, K = idx.shape
        batch = torch.arange(B, device=pcd_dino.device).unsqueeze(1).expand(B, K)  # [B,K]
        sampled_pcd_feat = pcd_dino[batch, idx, :]   # -> [B, K, C]
        sampled_pcd_rgb  = pcd_rgb[batch, idx, :]
        
        return sampled_pcd_coord, sampled_pcd_feat, sampled_pcd_rgb
    
    
    # ---------------- Orchestrator ----------------
    def run(self, pcd_data, batch_idx: int = 0,show_mask_pc: bool = False, show_pca_pc: bool = False) -> Dict[str, np.ndarray]:
        self._read_rgbd_and_calib(pcd_data, batch_idx=batch_idx)
        self._build_obs()
        self.build_point_cloud()
        self.run_fusion()
        pts, out = self.eval_vertices()
        rgb = out["color_tensor"][out["valid_mask"]]   # [N_valid, 3]

        vis_list = []
        if show_mask_pc:
            pc_mask = self.to_mask_colors(pts, out["mask"], out.get("valid_mask", None))
            vis_list.append(pc_mask)
        if show_pca_pc:
            pc_pca = self.pca_colors(pts, out["dino_feats"])
            vis_list.append(pc_pca)

        if vis_list:
            o3d.visualization.draw_geometries(vis_list)
            
        mask_onehot = onehot2instance(out['mask'])
        mask_onehot = mask_onehot[out['valid_mask']]
        pts = torch.from_numpy(pts).to(mask_onehot.device)
        pts = pts[out['valid_mask']]
        pcd_background = pts[mask_onehot==0]
        pcd_background_dino = out['dino_feats'][out['valid_mask']][mask_onehot==0]
        pcd_target = pts[mask_onehot!=0]
        pcd_target_dino = out['dino_feats'][out['valid_mask']][mask_onehot!=0]
        
        pcd_background_rgb = rgb[mask_onehot == 0]
        pcd_target_rgb = rgb[mask_onehot != 0]
        
        
        sampled_pcd_background, sampled_pcd_background_dino, sampled_pcd_background_rgb = self.pcd_background_fps(pcd_background, pcd_background_dino, pcd_background_rgb)

        sampled_pcd_target, sampled_pcd_target_dino, sampled_pcd_target_rgb = self.pcd_target_fps(pcd_target, pcd_target_dino, pcd_target_rgb)
        
        device = sampled_pcd_background.device
        sampled_pcd_background_dino = sampled_pcd_background_dino.to(device)
        sampled_pcd_target  = sampled_pcd_target.to(device)
        sampled_pcd_target_dino  = sampled_pcd_target_dino.to(device)

        xyz_all  = torch.cat([sampled_pcd_background,  sampled_pcd_target], dim=1)    # (B, Nb+Nt, 3)
        rgb_all = torch.cat([sampled_pcd_background_rgb, sampled_pcd_target_rgb], dim=1)  # [1, Nb+Nt, 3]


        feat_all = torch.cat([sampled_pcd_background_dino, sampled_pcd_target_dino], dim=1)    # (B, Nb+Nt, C)
        

        return {
            "pcd_coord": xyz_all,                 
            "pcd_dino": feat_all
        }
    
    def pad_point_batches(self, batch_xyz, batch_feat, pad_value=0.0):
        
        xyz_list  = [x.squeeze(0) for x in batch_xyz]   # [Ni, 3]
        feat_list = [f.squeeze(0) for f in batch_feat]  # [Ni, C]

        device = xyz_list[0].device

        coords = pad_sequence(xyz_list,  batch_first=True, padding_value=pad_value).to(device)  # [B, Nmax, 3]
        feats  = pad_sequence(feat_list, batch_first=True, padding_value=pad_value).to(device)  # [B, Nmax, C]

        lengths = torch.tensor([x.size(0) for x in xyz_list], device=device)                  # [B]
        Nmax = coords.size(1)
        mask = torch.arange(Nmax, device=device).unsqueeze(0) < lengths.unsqueeze(1)         # [B, Nmax]
        
        return coords, feats, mask
        
    def run_batch(self, pcd_data,
              show_mask_pc: bool = False, show_pca_pc: bool = False, mode = None) -> Dict[str, torch.Tensor]:
        B = pcd_data['color'].shape[0]
        batch_xyz = []
        batch_feat = []
        for b in range(B):
            out_b = self.run(pcd_data, batch_idx=b,
                            show_mask_pc=show_mask_pc, show_pca_pc=show_pca_pc)
            batch_xyz.append(out_b["pcd_coord"])  # [1, P, 3]
            batch_feat.append(out_b["pcd_dino"])  # [1, P, C]

        pcd_coord, pcd_dino, pcd_batch_mask = self.pad_point_batches(batch_xyz, batch_feat)
        # pcd_coord = torch.cat(batch_xyz, dim=0)   # [B, P, 3]
        # pcd_dino  = torch.cat(batch_feat, dim=0)  # [B, P, C]
        torch.cuda.empty_cache()
        return {"pcd_coord": pcd_coord, "pcd_dino": pcd_dino, "pcd_batch_mask": pcd_batch_mask}

if __name__ == "__main__":
    cfg = PcdEncoderCfg(
        num_cam=1,
        device="cuda",
        pointcloud_encoder_cfg_backgroud={
            "in_channels" : 1027,
            "out_channels" : 2048, 
            "npoint" : 512,
            "use_bn" : True
        },
        pointcloud_encoder_cfg_target={
            "in_channels" : 1027,
            "out_channels" : 2048, 
            "npoint" : 2048,
            "use_bn" : True
        }
    )
    runner = PcdEncoder(cfg)
    results = runner.run(show_raw=False, show_pca=True)

    pts = results["points"]
    mask_inst = results["mask_instance"]
    pcd_background = pts[mask_inst == 0]
    pcd_target = pts[mask_inst != 0]
