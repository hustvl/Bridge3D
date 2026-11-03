import re, sys, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from pathlib import Path
# get current workspace
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(current_file.parent))
sys.path.append(str(project_root))
from hub_mixin import CompatiblePyTorchModelHubMixin
from rdt.model import RDT

from third_party.vggt.models.vggt import VGGT


from models.pcdencoder.PcdEncoder import PcdEncoder, PcdEncoderCfg

# from policy.Bridge3D.models.pcdencoder.PcdEncoder import PcdEncoder, PcdEncoderCfg

from PIL import Image
from torchvision import transforms as TF


class VGGTProjector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VGGTProjector, self).__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim, eps=1e-6),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)

class RDTRunner(nn.Module,
                CompatiblePyTorchModelHubMixin,
                repo_url="https://huggingface.co/robotics-diffusion-transformer/rdt-1b"):

    def __init__(self,
                 *,
                 action_dim,
                 pred_horizon,
                 config,
                 lang_token_dim,
                 img_token_dim,
                 state_token_dim,
                 max_lang_cond_len,
                 img_cond_len,
                 lang_pos_embed_config=None,
                 img_pos_embed_config=None,
                 dtype=torch.bfloat16,
                 use_vggt=False):
        super(RDTRunner, self).__init__()
        # Create diffusion model
        hidden_size = config['rdt']['hidden_size']
        self.model = RDT(
            output_dim=action_dim,
            horizon=pred_horizon,
            hidden_size=hidden_size,
            depth=config['rdt']['depth'],
            num_heads=config['rdt']['num_heads'],
            max_lang_cond_len=max_lang_cond_len,
            img_cond_len=img_cond_len,
            lang_pos_embed_config=lang_pos_embed_config,
            img_pos_embed_config=img_pos_embed_config,
            dtype=dtype,
        )
        if use_vggt:
            # Setting VGGT
            self.vggt = VGGT.from_pretrained("facebook/VGGT-1B")
            del self.vggt.camera_head
            del self.vggt.point_head
            del self.vggt.depth_head
            del self.vggt.track_head
            torch.cuda.empty_cache()

            for param in self.vggt.parameters():
                param.requires_grad = False

            self.vggtprojector = VGGTProjector(input_dim=2048, 
                                hidden_dim=1600, output_dim=config['img_token_dim'])

        # Create adpators for various conditional inputs
        self.lang_adaptor = self.build_condition_adapter(config['lang_adaptor'],
                                                         in_features=lang_token_dim,
                                                         out_features=hidden_size)
        self.img_adaptor = self.build_condition_adapter(config['img_adaptor'],
                                                        in_features=img_token_dim,
                                                        out_features=hidden_size)
        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config['state_adaptor'],
            in_features=state_token_dim * 2,  # state + state mask (indicator)
            out_features=hidden_size)

        # Create the noise scheduler
        noise_scheduler_config = config['noise_scheduler']
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
            clip_sample=noise_scheduler_config['clip_sample'],
        )
        self.noise_scheduler_sample = DPMSolverMultistepScheduler(
            num_train_timesteps=noise_scheduler_config['num_train_timesteps'],
            beta_schedule=noise_scheduler_config['beta_schedule'],
            prediction_type=noise_scheduler_config['prediction_type'],
        )

        self.num_train_timesteps = noise_scheduler_config['num_train_timesteps']
        self.num_inference_timesteps = noise_scheduler_config['num_inference_timesteps']
        self.prediction_type = noise_scheduler_config['prediction_type']

        self.pred_horizon = pred_horizon
        self.action_dim = action_dim

        # PcdEncoder

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        pcdencoder_cfg = PcdEncoderCfg(
            num_cam=1,
            device=f"cuda:{local_rank}",
            pointcloud_encoder_cfg_backgroud={
                "in_channels" : 1027,
                "out_channels" : 1021, 
                "use_bn" : True
            },
            pointcloud_encoder_cfg_target={
                "in_channels" : 1027,
                "out_channels" : 1021, 
                "use_bn" : True
            }
        )
        self.PcdEncoder = PcdEncoder(pcdencoder_cfg)

        print("Diffusion params: %e" %
              sum([p.numel() for p in self.model.parameters()] + [p.numel() for p in self.lang_adaptor.parameters()] +
                  [p.numel()
                   for p in self.img_adaptor.parameters()] + [p.numel() for p in self.state_adaptor.parameters()]))

    def build_condition_adapter(self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector

    def adapt_conditions(self, lang_tokens, img_tokens, state_tokens):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, state_len, state_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_lang = self.lang_adaptor(lang_tokens)
        adpated_img = self.img_adaptor(img_tokens)
        adpated_state = self.state_adaptor(state_tokens)

        return adpated_lang, adpated_img, adpated_state

    def conditional_sample(self, lang_cond, lang_attn_mask, img_cond, state_traj, action_mask, ctrl_freqs, pcd_input=None):
        '''
        lang_cond: language conditional data, (batch_size, lang_len, hidden_size).
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_cond: image conditional data, (batch_size, img_len, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        action_mask: (batch_size, 1, action_dim), a 0-1 **float** tensor
            indicating the valid action dimensions.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        noisy_action = torch.randn(size=(state_traj.shape[0], self.pred_horizon, self.action_dim),
                                   dtype=dtype,
                                   device=device)
        action_mask = action_mask.expand(-1, self.pred_horizon, -1)

        # Set step values
        self.noise_scheduler_sample.set_timesteps(self.num_inference_timesteps)

        print("[RDT] before PcdEncoder:  ",
          torch.cuda.memory_allocated(device)/1024**2, "MB")

        if pcd_input is not None:
            with torch.no_grad():
                pcd_output = self.PcdEncoder.run_batch(pcd_input, show_mask_pc=False, show_pca_pc=False)

        print("[RDT] after PcdEncoder:   ",
          torch.cuda.memory_allocated(device)/1024**2, "MB")

        for t in self.noise_scheduler_sample.timesteps:
            # Prepare state-action trajectory
            action_traj = torch.cat([noisy_action, action_mask], dim=2)
            action_traj = self.state_adaptor(action_traj)
            state_action_traj = torch.cat([state_traj, action_traj], dim=1)

            # Predict the model output
            model_output = self.model(state_action_traj,
                                      ctrl_freqs,
                                      t.unsqueeze(-1).to(device),
                                      lang_cond,
                                      img_cond,
                                      lang_mask=lang_attn_mask,
                                      pcd_input=pcd_input, 
                                      pcd_output=pcd_output)

            # Compute previous actions: x_t -> x_t-1
            noisy_action = self.noise_scheduler_sample.step(model_output, t, noisy_action).prev_sample
            noisy_action = noisy_action.to(state_traj.dtype)

        print("[RDT] after model:        ",
          torch.cuda.memory_allocated(device)/1024**2, "MB")

        # Finally apply the action mask to mask invalid action dimensions
        noisy_action = noisy_action * action_mask

        return noisy_action
    
    def load_and_preprocess_images(self, images_list, mode="pad"):
        """
        A quick start function to load and preprocess images for model input.
        This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

        Args:
            image_path_list (list): List of paths to image files
            mode (str, optional): Preprocessing mode, either "crop" or "pad".
                                - "crop" (default): Sets width to 518px and center crops height if needed.
                                - "pad": Preserves all pixels by making the largest dimension 518px
                                and padding the smaller dimension to reach a square shape.

        Returns:
            torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

        Raises:
            ValueError: If the input list is empty or if mode is invalid

        Notes:
            - Images with different dimensions will be padded with white (value=1.0)
            - A warning is printed when images have different shapes
            - When mode="crop": The function ensures width=518px while maintaining aspect ratio
            and height is center-cropped if larger than 518px
            - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
            and the smaller dimension is padded to reach a square shape (518x518)
            - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
        """
        # Validate mode
        if mode not in ["crop", "pad"]:
            raise ValueError("Mode must be either 'crop' or 'pad'")

        images = []
        shapes = set()
        to_tensor = TF.ToTensor()
        target_size = 518

        # First process all images and collect their shapes
        for img in images_list:

            img = img.convert("RGB")

            width, height = img.size

            if mode == "pad":
                # Make the largest dimension 518px while maintaining aspect ratio
                if width >= height:
                    new_width = target_size
                    new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
                else:
                    new_height = target_size
                    new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
            else:  # mode == "crop"
                # Original behavior: set width to 518px
                new_width = target_size
                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width) / 14) * 14

            # Resize with new dimensions (width, height)
            img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
            img = to_tensor(img)  # Convert to tensor (0, 1)

            # Center crop height if it's larger than 518 (only in crop mode)
            if mode == "crop" and new_height > target_size:
                start_y = (new_height - target_size) // 2
                img = img[:, start_y : start_y + target_size, :]

            # For pad mode, pad to make a square of target_size x target_size
            if mode == "pad":
                h_padding = target_size - img.shape[1]
                w_padding = target_size - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    # Pad with white (value=1.0)
                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )

            shapes.add((img.shape[1], img.shape[2]))
            images.append(img)

        # Check if we have different shapes
        # In theory our model can also work well with different shapes
        if len(shapes) > 1:
            print(f"Warning: Found images with different shapes: {shapes}")
            # Find maximum dimensions
            max_height = max(shape[0] for shape in shapes)
            max_width = max(shape[1] for shape in shapes)

            # Pad images if necessary
            padded_images = []
            for img in images:
                h_padding = max_height - img.shape[1]
                w_padding = max_width - img.shape[2]

                if h_padding > 0 or w_padding > 0:
                    pad_top = h_padding // 2
                    pad_bottom = h_padding - pad_top
                    pad_left = w_padding // 2
                    pad_right = w_padding - pad_left

                    img = torch.nn.functional.pad(
                        img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                    )
                padded_images.append(img)
            images = padded_images

        images = torch.stack(images)  # concatenate images

        # Ensure correct shape when single image
        if len(images_list) == 1:
            # Verify shape is (1, C, H, W)
            if images.dim() == 3:
                images = images.unsqueeze(0)

        return images


    # ========= Train  ============
    def compute_loss(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_gt, action_mask,
                     ctrl_freqs, images_vggt_list, pcd_input, use_vggt) -> torch.Tensor:
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)  torch.Size([batch_size, 6*729, 1152])
        state_tokens: (batch_size, 1, state_token_dim)
        action_gt: (batch_size, horizon, state_token_dim), ground-truth actions for supervision
        action_mask: (batch_size, 1, state_token_dim), a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: loss_value, a scalar tensor
        '''
        if use_vggt:
            
            '''
                ---Get vggt image features---
            '''
            B = len(images_vggt_list)
            S = len(images_vggt_list[0])
            flat_images = [img for sample_imgs in images_vggt_list for img in sample_imgs]
            # len(flat_images) = B * S

            images_flat = self.load_and_preprocess_images(flat_images, mode="pad").to(action_gt.device) # torch.Size([B*S, 3, 518, 518])
            C, H, W = images_flat.shape[1:]
            images_vggt = images_flat.view(B*S, 1, C, H, W)

            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images_vggt) # len(aggregated_tokens_list)=24
            tokens = aggregated_tokens_list[-2]             # -> [B*S, P', 2048]
            tokens = tokens[:, 0]                    # [B*S, P, 2*D]
            # 3) keep only patch tokens
            vggt_patch_tokens = tokens[:, patch_start_idx:]           # -> [B*S, num_patch, 2048]  torch.Size([24, 1369, 2048])
            H_patch = int(vggt_patch_tokens.shape[1]**0.5)
            vggt_feat = vggt_patch_tokens.view(B * S, H_patch, H_patch, 2048).permute(0, 3, 1, 2)   # [B*S, 2048, 37, 37]
            vggt_feat_resized = F.interpolate(vggt_feat, size=(27, 27),
                                    mode="bilinear", align_corners=False)  # [B*S, 2048, 27, 27]
            vggt_patches_resized = vggt_feat_resized.permute(0, 2, 3, 1).reshape(B, S, 27, 27, 2048)  # [B, S, 27, 27, 2048]

            vggt_patch_feats = vggt_patches_resized.reshape(B, S*27*27, 2048).to(torch.bfloat16)

            img_vggt_tokens = self.vggtprojector(vggt_patch_feats)
            
            '''
                Add
            '''
            img_tokens = img_tokens + img_vggt_tokens
            '''
                Concat
            '''
            # img_tokens = torch.cat([img_tokens, img_vggt_tokens], dim=1)

            '''
                ---Get vggt image features---
            '''

        if pcd_input is not None:
            with torch.no_grad():
                pcd_output = self.PcdEncoder.run_batch(pcd_input, show_mask_pc=False, show_pca_pc=False)


        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        # Sample noise that we'll add to the actions
        noise = torch.randn(action_gt.shape, dtype=action_gt.dtype, device=device)
        # Sample random diffusion timesteps
        timesteps = torch.randint(0, self.num_train_timesteps, (batch_size, ), device=device).long()
        # Add noise to the clean actions according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_action = self.noise_scheduler.add_noise(action_gt, noise, timesteps)

        # Concatenate the state and action tokens to form the input sequence
        state_action_traj = torch.cat([state_tokens, noisy_action], dim=1)
        # Append the action mask to the input sequence
        action_mask = action_mask.expand(-1, state_action_traj.shape[1], -1)
        state_action_traj = torch.cat([state_action_traj, action_mask], dim=2)
        # Align the dimension with the hidden size
        lang_cond, img_cond, state_action_traj = self.adapt_conditions(lang_tokens, img_tokens, state_action_traj)
        # Predict the denoised result
        # pred, ee_pred_list, ee_xyz_target_list = self.model(state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, lang_mask=lang_attn_mask, 
        #                   pcd_input=pcd_input, pcd_output=pcd_output)
        pred = self.model(state_action_traj, ctrl_freqs, timesteps, lang_cond, img_cond, lang_mask=lang_attn_mask, 
                          pcd_input=pcd_input, pcd_output=pcd_output)

        pred_type = self.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = F.mse_loss(pred, target)
        
            
        return loss

    # ========= Inference  ============
    def predict_action(self, lang_tokens, lang_attn_mask, img_tokens, state_tokens, action_mask, ctrl_freqs, pcd_input=None):
        '''
        lang_tokens: (batch_size, lang_len, lang_token_dim)
        lang_attn_mask: (batch_size, lang_len), a mask for valid language tokens,
            which should be True-False bool tensor.
        img_tokens: (batch_size, img_len, img_token_dim)
        state_tokens: (batch_size, 1, state_token_dim)
        action_mask: (batch_size, 1, action_dim),
            which should be a 0-1 **float** tensor.
        ctrl_freqs: (batch_size,), control frequency for each sample.
        
        return: (batch_size, horizon, action_dim), predicted action sequence
        '''
        # Prepare the state and conditions
        state_tokens = torch.cat([state_tokens, action_mask], dim=2)
        lang_cond, img_cond, state_traj = self.adapt_conditions(lang_tokens, img_tokens, state_tokens)

        # Run sampling
        action_pred = self.conditional_sample(
            lang_cond,
            lang_attn_mask,
            img_cond,
            state_traj,
            action_mask,
            ctrl_freqs,
            pcd_input=pcd_input
        )

        return action_pred

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.compute_loss(*args, **kwargs)
