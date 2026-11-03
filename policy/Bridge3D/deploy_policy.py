# import packages and module here
import sys, os
from .model import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

from typing import List, Literal

import pybullet as p
import pybullet_data
import transforms3d as t3d
import os

TARGETS = np.array([[0, 100, 0],
                    [0, 128, 0]], dtype=np.uint8)

def rgb_to_onehot(mask_hw3: np.ndarray) -> np.ndarray:
    """(H, W, 3) -> (H, W, 2) onehot"""
    hit_any = np.any(np.all(mask_hw3[..., None, :] == TARGETS, axis=-1), axis=-1)
    labels = np.where(hit_any, 0, 1).astype(np.int64)
    onehot = np.eye(2, dtype=np.uint8)[labels]
    return onehot



def encode_obs(observation, pcd_input=None):  # Post-Process Observation
    observation["agent_pos"] = observation["joint_action"]["vector"]

    if pcd_input:
        pcd_mask = np.expand_dims(rgb_to_onehot(observation['observation']['head_camera']['actor_segmentation']), axis=0)
        observation['pcd_mask'] = pcd_mask
        depth = observation['observation']['head_camera']['depth'] / 1000.0
        observation['depth'] = np.expand_dims(depth, axis=0)
        extrinsics = np.expand_dims(observation['observation']['head_camera']['extrinsic_cv'], axis=0)
        observation['extrinsics'] = extrinsics[np.newaxis, ...]
        K = observation['observation']['head_camera']['intrinsic_cv']
        intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        observation['intrinsics'] = intrinsics[np.newaxis, np.newaxis, ...]
        state_left_arm = np.expand_dims(observation['joint_action']["left_arm"], axis=0)
        observation['state_left_arm'] = state_left_arm[np.newaxis, ...]
        state_right_arm = np.expand_dims(observation['joint_action']["right_arm"], axis=0)
        observation['state_right_arm'] = state_right_arm[np.newaxis, ...]
        color = observation['observation']['head_camera']['rgb']
        observation['color'] = np.expand_dims(color, axis=0)

    return observation


def get_model(usr_args, fk_engine=None):  # keep
    print("----usr_args.keys():----", usr_args.keys())
    print("----usr_args['use_vggt']:----", usr_args['use_vggt'])
    model_name = usr_args["ckpt_setting"]
    checkpoint_id = usr_args["checkpoint_id"]
    left_arm_dim, right_arm_dim, rdt_step = (
        usr_args["left_arm_dim"],
        usr_args["right_arm_dim"],
        usr_args["rdt_step"],
    )
    rdt = RDT(
        os.path.join(
            parent_directory,
            f"checkpoints/{model_name}/checkpoint-{checkpoint_id}/pytorch_model/mp_rank_00_model_states.pt",
        ),
        usr_args["task_name"],
        left_arm_dim,
        right_arm_dim,
        rdt_step,
        use_vggt=usr_args['use_vggt']
    )
    
    rdt.policy.policy.model.fk_engine = fk_engine
    return rdt


def eval(task_name, TASK_ENV, model, observation, temporal_agg=False, use_vggt=False, pcd_input=False):
    """x
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation, pcd_input=True)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()
    input_rgb_arr, input_state = [
        obs["observation"]["head_camera"]["rgb"],
        obs["observation"]["right_camera"]["rgb"],
        obs["observation"]["left_camera"]["rgb"],
    ], obs["agent_pos"]  # TODO

    pcd_input = dict()
    pcd_input['depth'] = obs['depth']
    pcd_input['pcd_mask'] = obs['pcd_mask']
    pcd_input['extrinsics'] = obs['extrinsics']
    pcd_input['intrinsics'] = obs['intrinsics']
    pcd_input['left_arm_state'] = obs['state_left_arm']
    pcd_input['right_arm_state'] = obs['state_right_arm']
    pcd_input['color'] = obs['color']

    if (model.observation_window
            is None):  # Force an update of the observation at the first frame to avoid an empty observation window
        model.set_language_instruction(instruction)
        model.update_observation_window(input_rgb_arr, input_state)
        
    import time
    torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    actions = model.get_action(pcd_input=pcd_input, use_vggt=use_vggt)  # Get Action according to observation chunk
    
    torch.cuda.synchronize()
    
    inference_time = time.perf_counter() - start_time
    
    print(f"Inference time:{inference_time*1000:3f} ms")

    for action in actions[:16]:  # Execute each step of the action
        TASK_ENV.take_action(action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        input_rgb_arr, input_state = [
            obs["observation"]["head_camera"]["rgb"],
            obs["observation"]["right_camera"]["rgb"],
            obs["observation"]["left_camera"]["rgb"],
        ], obs["agent_pos"]  # TODO
        model.update_observation_window(input_rgb_arr, input_state)  # Update Observation


def reset_model(
        model):  # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    model.reset_obsrvationwindows()
    torch.cuda.empty_cache()
