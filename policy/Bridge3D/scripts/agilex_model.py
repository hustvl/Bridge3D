import os, sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING

from pathlib import Path

# get current workspace
current_file = Path(__file__)
sys.path.append(os.path.join(current_file.parent.parent, "models"))

from multimodal_encoder.siglip_encoder import SiglipVisionTower
from multimodal_encoder.t5_encoder import T5Embedder
from rdt_runner import RDTRunner

from PIL import Image
from torchvision import transforms as TF
import torch.nn.functional as F
# The indices that the raw vector should be mapped to in the unified action vector
# AGILEX_STATE_INDICES = [
#     STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(1)
# ] + [
#     STATE_VEC_IDX_MAPPING["left_gripper_open"]
# ] + [
#     STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(1)
# ] + [
#     STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
# ]
# AGILEX_STATE_INDICES = None


# Create the RDT model
def create_model(args, **kwargs):
    left_arm_dim, right_arm_dim = (
        args["arm_dim"]["left_arm_dim"],
        args["arm_dim"]["right_arm_dim"],
    )
    AGILEX_STATE_INDICES = ([STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                             for i in range(left_arm_dim)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                            [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                             for i in range(right_arm_dim)] + [STATE_VEC_IDX_MAPPING[f"right_gripper_open"]])
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if pretrained is not None and os.path.isfile(pretrained):
        model.load_pretrained_weights(pretrained)

    return model


class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
    1. Model initialization
    2. Encodings of instructions
    3. Model inference
    """

    def __init__(
        self,
        args,
        device="cuda",
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=25,
        pretrained=None,
        pretrained_vision_encoder_name_or_path=None,
        use_vggt=False
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        # We do not use the text encoder due to limited GPU memory
        # self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy(pretrained, use_vggt=use_vggt)
        self.left_arm_dim, self.right_arm_dim = (
            args["arm_dim"]["left_arm_dim"],
            args["arm_dim"]["right_arm_dim"],
        )

        self.reset()

    def get_policy(self, pretrained, use_vggt=False):
        """Initialize the model."""
        # Initialize model with arguments
        if pretrained is None or os.path.isfile(pretrained):
            img_cond_len = (self.args["common"]["img_history_size"] * self.args["common"]["num_cameras"] *
                            self.vision_model.num_patches)

            _model = RDTRunner(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    # No initial pos embed in the last grid size
                    # since we've already done in ViT
                    (
                        "image",
                        (
                            self.args["common"]["img_history_size"],
                            self.args["common"]["num_cameras"],
                            -self.vision_model.num_patches,
                        ),
                    ),
                ],
                lang_pos_embed_config=[
                    # Similarly, no initial pos embed for language
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
                use_vggt=use_vggt
            )
        else:
            _model = RDTRunner.from_pretrained(pretrained, use_vggt=use_vggt)

        return _model

    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(
            from_pretrained=pretrained_text_encoder_name_or_path,
            model_max_length=self.args["dataset"]["tokenizer_max_length"],
            device=self.device,
        )
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    def reset(self):
        """Set model to evaluation mode."""
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        # self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        # self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    # def load_pretrained_weights(self, pretrained=None):
    #     if pretrained is None:
    #         return
    #     print(f"Loading weights from {pretrained}")
    #     filename = os.path.basename(pretrained)
    #     if filename.endswith(".pt"):
    #         checkpoint = torch.load(pretrained)
    #         self.policy.load_state_dict(checkpoint["module"])
    #     elif filename.endswith(".safetensors"):
    #         from safetensors.torch import load_model

    #         load_model(self.policy, pretrained)
    #     else:
    #         raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")
    
    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return

        print(f"Loading weights from {pretrained}")
        filename = os.path.basename(pretrained)

        if filename.endswith(".pt"):
            checkpoint = torch.load(pretrained, map_location="cpu")
            checkpoint_state = checkpoint["module"]

            model_state = self.policy.state_dict()

            loadable_state = {}
            skipped_missing = []
            skipped_mismatch = []

            for key, value in checkpoint_state.items():
                if key not in model_state:
                    skipped_missing.append(key)
                    continue

                if value.shape != model_state[key].shape:
                    skipped_mismatch.append((key, value.shape, model_state[key].shape))
                    continue

                loadable_state[key] = value

            model_state.update(loadable_state)

            self.policy.load_state_dict(model_state, strict=True)

            current_keys = set(model_state.keys())
            loaded_keys = set(loadable_state.keys())
            newly_initialized = sorted(current_keys - loaded_keys)

            print(f"Loaded {len(loadable_state)} tensors from checkpoint.")
            print(f"Kept initialized {len(newly_initialized)} tensors not loaded from checkpoint.")

            if skipped_mismatch:
                print("Skipped shape-mismatched tensors:")
                for key, ckpt_shape, model_shape in skipped_mismatch:
                    print(f"  {key}: checkpoint {tuple(ckpt_shape)} -> model {tuple(model_shape)}")

        elif filename.endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint_state = load_file(pretrained)
            model_state = self.policy.state_dict()

            loadable_state = {}
            skipped_mismatch = []

            for key, value in checkpoint_state.items():
                if key not in model_state:
                    continue

                if value.shape != model_state[key].shape:
                    skipped_mismatch.append((key, value.shape, model_state[key].shape))
                    continue

                loadable_state[key] = value

            model_state.update(loadable_state)
            self.policy.load_state_dict(model_state, strict=True)

            print(f"Loaded {len(loadable_state)} tensors from safetensors.")
            print(f"Kept initialized {len(model_state) - len(loadable_state)} tensors not loaded from checkpoint.")

            if skipped_mismatch:
                print("Skipped shape-mismatched tensors:")
                for key, ckpt_shape, model_shape in skipped_mismatch:
                    print(f"  {key}: checkpoint {tuple(ckpt_shape)} -> model {tuple(model_shape)}")

        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    def encode_instruction(self, instruction, device="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device

        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(instruction, return_tensors="pt", padding="longest",
                                     truncation=True)["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()

        return pred

    def _format_joint_to_state(self, joints):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted.
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]).
        """
        AGILEX_STATE_INDICES = ([STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                                 for i in range(self.left_arm_dim)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                                [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                                 for i in range(self.right_arm_dim)] + [STATE_VEC_IDX_MAPPING[f"right_gripper_open"]])
        # Rescale the gripper to the range of [0, 1]
        joints = joints / torch.tensor(
            [[[1 for i in range(self.left_arm_dim + 1 + self.right_arm_dim + 1)]]],
            device=joints.device,
            dtype=joints.dtype,
        )

        B, N, _ = joints.shape
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        # Fill into the unified state vector
        state[:, :, AGILEX_STATE_INDICES] = joints
        # Assemble the mask indicating each dimension's availability
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device,
            dtype=joints.dtype,
        )
        state_elem_mask[:, AGILEX_STATE_INDICES] = 1
        return state, state_elem_mask

    def _unformat_action_to_joint(self, action):
        """
        Unformat the unified action vector into the joint action to be executed.

        Args:
            action (torch.Tensor): The unified action vector to be unformatted.
                ([B, N, 128])

        Returns:
            joints (torch.Tensor): The unformatted robot joint action.
                qpos ([B, N, 14]).
        """
        AGILEX_STATE_INDICES = ([STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"]
                                 for i in range(self.left_arm_dim)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]] +
                                [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"]
                                 for i in range(self.right_arm_dim)] + [STATE_VEC_IDX_MAPPING[f"right_gripper_open"]])
        action_indices = AGILEX_STATE_INDICES
        joints = action[:, :, action_indices]

        # Rescale the gripper back to the action range
        # Note that the action range and proprioception range are different
        # for Mobile ALOHA robot
        joints = joints * torch.tensor(
            [[[1 for i in range(self.left_arm_dim + 1 + self.right_arm_dim + 1)]]],
            device=joints.device,
            dtype=joints.dtype,
        )

        return joints
    
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
    
    def process_imgs_vggt(self, images_vggt_list, device):
        B = 1
        S = len(images_vggt_list)
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        flat_images = images_vggt_list
        # len(flat_images) = B * S

        images_flat = self.load_and_preprocess_images(flat_images, mode="pad").to(device=device, dtype=dtype) # torch.Size([B*S, 3, 518, 518])
        C, H, W = images_flat.shape[1:]
        images_vggt = images_flat.view(B*S, 1, C, H, W)

        N = images_vggt.shape[0]
        chunk_size = int(B*S / 2)

        all_tokens = []

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                images_chunk = images_vggt[start:end]      # [chunk, 1, C, H, W]

                with torch.cuda.amp.autocast(dtype=dtype):
                    aggregated_tokens_list, patch_start_idx = self.policy.vggt.aggregator(images_chunk)

                tokens_chunk = aggregated_tokens_list[-2]

                all_tokens.append(tokens_chunk[:, 0])

                del aggregated_tokens_list, tokens_chunk, images_chunk
                torch.cuda.empty_cache()

            tokens = torch.cat(all_tokens, dim=0)

        vggt_patch_tokens = tokens[:, patch_start_idx:]           # -> [B*S, num_patch, 2048]  torch.Size([24, 1369, 2048])
        H_patch = int(vggt_patch_tokens.shape[1]**0.5)
        vggt_feat = vggt_patch_tokens.view(B * S, H_patch, H_patch, 2048).permute(0, 3, 1, 2)   # [B*S, 2048, 37, 37]
        vggt_feat_resized = F.interpolate(vggt_feat, size=(27, 27),
                                  mode="bilinear", align_corners=False)  # [B*S, 2048, 27, 27]
        vggt_patches_resized = vggt_feat_resized.permute(0, 2, 3, 1).reshape(B, S, 27, 27, 2048)  # [B, S, 27, 27, 2048]

        vggt_patch_feats = vggt_patches_resized.reshape(B, S*27*27, 2048).to(torch.bfloat16)

        img_vggt_tokens = self.policy.vggtprojector(vggt_patch_feats)

        return img_vggt_tokens

    @torch.no_grad()
    def step(self, proprio, images, text_embeds, pcd_input=None, use_vggt=False):
        """
        Predict the next action chunk given the
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1},
                ext_{t}, right_wrist_{t}, left_wrist_{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype

        # The background image used for padding
        background_color = np.array([int(x * 255) for x in self.image_processor.image_mean],
                                    dtype=np.uint8).reshape(1, 1, 3)
        background_image = (np.ones(
            (
                self.image_processor.size["height"],
                self.image_processor.size["width"],
                3,
            ),
            dtype=np.uint8,
        ) * background_color)

        # Preprocess the images by order and encode them
        image_tensor_list = []
        images_vggt_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)

            if use_vggt:
                images_vggt_list.append(image.copy())    

            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)

            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75, 1.75))(image)

            if self.args["dataset"].get("image_aspect_ratio", "pad") == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                image = expand2square(image, tuple(int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            image_tensor_list.append(image)

        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)
        if use_vggt:
            import time
            torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            image_vggt_embeds = self.process_imgs_vggt(images_vggt_list, device)
            image_embeds = image_embeds + image_vggt_embeds
            
            torch.cuda.synchronize()
    
            vggt_time = time.perf_counter() - start_time
        
            print(f"vggt time:{vggt_time*1000:3f} ms")
            
        # Prepare the proprioception states and the control frequency
        joints = proprio.to(device).unsqueeze(0)  # (1, 1, 14)
        states, state_elem_mask = self._format_joint_to_state(joints)  # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)

        text_embeds = text_embeds.to(device, dtype=dtype)

        # Predict the next action chunk given the inputs
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(text_embeds.shape[:2], dtype=torch.bool, device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),
            ctrl_freqs=ctrl_freqs,
            pcd_input=pcd_input
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory

