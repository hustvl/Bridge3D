<div align="center">

<h1>Bridge3D: Enabling Vision–Language–Action Models<br>to See and Act in 3D</h1>

<p>
  Haoxuan Li<sup>1,*</sup>, Sixu Yan<sup>1,*</sup>, Lianghui Zhu<sup>1,*</sup>,
  Xuanlai Tang<sup>2</sup>, Shikang Wang<sup>1</sup>, Xinggang Wang<sup>1,†</sup>
</p>

<p>
  <sup>1</sup>Huazhong University of Science and Technology &nbsp;
  <sup>2</sup>KEENON Robotics
</p>

<p><sup>*</sup>Equal contribution &nbsp; <sup>†</sup>Corresponding author</p>

<p>
  <a href="https://arxiv.org/abs/2609.24525"><img src="https://img.shields.io/badge/Paper-arXiv-b31b1b" alt="Paper"></a>
  <a href="https://hustvl.github.io/Bridge3D/"><img src="https://img.shields.io/badge/Project-Page-008080" alt="Project Page"></a>
  <a href="https://github.com/hustvl/Bridge3D"><img src="https://img.shields.io/badge/Code-GitHub-181717" alt="Code"></a>
</p>

</div>

Bridge3D brings 3D spatial information into pretrained vision–language–action models. It combines **Implicit Fusion**, which incorporates geometry features into visual tokens, with **Explicit Conditioning**, which supplies a 3D semantic field to the action model. This release contains the RDT-based implementation for training and evaluation on RoboTwin 2.0.

This repository provides the RDT-Bridge3D policy implementation and supporting scripts. To run the complete pipeline, integrate the provided code into the official RoboTwin codebase following the RoboTwin policy deployment guide.
## Setup

### Installation


```bash
git clone --recurse-submodules https://github.com/hustvl/Bridge3D.git
cd Bridge3D
pip install -r policy/Bridge3D/requirements.txt
```

### Pretrained Models

Download the RDT backbone, text encoder, and vision encoder:

```bash
mkdir -p policy/weights/RDT

huggingface-cli download robotics-diffusion-transformer/rdt-1b \
  --local-dir policy/weights/RDT/rdt-1b
huggingface-cli download google/t5-v1_1-xxl \
  --local-dir policy/weights/RDT/t5-v1_1-xxl
huggingface-cli download google/siglip-so400m-patch14-384 \
  --local-dir policy/weights/RDT/siglip-so400m-patch14-384
```

### Data Preparation

Collect demonstrations and generate language instructions using the RoboTwin workflow. The processor expects the following layout under the repository root:

```text
data/
└── click_bell/
    └── demo_clean/
        ├── data/
        │   ├── episode0.hdf5
        │   └── ...
        └── instructions/
            ├── episode0.json
            └── ...
```

Each demonstration must include the head-camera `depth` and `actor_segmentation` observations, along with the three camera RGB streams and joint actions.

Switch to the policy directory for the remaining commands:

```bash
cd policy/Bridge3D
mkdir -p processed_data training_data
```

Set the following variables inside [`process_data_rdt_bridge3d.sh`](policy/Bridge3D/process_data_rdt_bridge3d.sh):

```bash
task_name=click_bell
task_config=demo_clean
expert_data_num=50
gpu_id=0
```

Then process the demonstrations:

```bash
bash process_data_rdt_bridge3d.sh
```

Copy the processed dataset into `training_data/`, then extract and save the camera parameters:

```bash
python scripts/extract_cam_params.py click_bell demo_clean 50
```

The processed `training_data/` directory is organized as follows:

```text
training_data/click_bell-demo_clean-50/
├── cam_params/
│   ├── camera_extrinsics.npy  # float32, (4, 4)
│   └── camera_params.npy      # float32, (4,), [fx, fy, cx, cy]
├── episode_0/
│   ├── episode_0.hdf5
│   └── instructions/
└── ...
```


## Usage

### Training

The example experiment is `click_bell_demo_clean_bridge3d`.

Update the corresponding fields in [`model_config/click_bell_demo_clean_bridge3d.yml`](policy/Bridge3D/model_config/click_bell_demo_clean_bridge3d.yml):

```yaml
model: click_bell_demo_clean_bridge3d
data_path: ./training_data/click_bell-demo_clean-50
checkpoint_path: ./checkpoints/click_bell_demo_clean_bridge3d
pretrained_model_name_or_path: ../weights/RDT/rdt-1b
cuda_visible_device: "0,1,2,3"
```
Launch training:

```bash
bash finetune.sh
```

### Evaluation

Set the task and checkpoint variables inside [`eval.sh`](policy/Bridge3D/eval.sh):

```bash
policy_name=Bridge3D
task_name=click_bell
task_config=demo_clean
model_name=click_bell_demo_clean_bridge3d
checkpoint_id=10000
seed=0
gpu_id=0
```

Keep `pcd_input: True` and `use_vggt: True` in [`deploy_policy.yml`](policy/Bridge3D/deploy_policy.yml) to evaluate both Bridge3D branches. Run evaluation:

```bash
bash eval.sh
```


## Acknowledgements

Bridge3D builds on [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer), [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin). We thank their authors for making the implementations available.

## Citation

If Bridge3D contributes to your research, please cite:

```bibtex
@article{li2026bridge3d,
  title={Bridge3D: Enabling Vision-Language-Action Models to See and Act in 3D},
  author={Li, Haoxuan and Yan, Sixu and Zhu, Lianghui and Tang, Xuanlai and Wang, Shikang and Wang, Xinggang},
  journal={arXiv preprint arXiv:2609.24525},
  year={2026},
  url={https://arxiv.org/abs/2609.24525}
}
```
