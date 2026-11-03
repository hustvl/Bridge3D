"""Export the shared camera calibration expected by the Bridge3D dataset loader.

Example (from policy/Bridge3D):
    python scripts/extract_cam_params.py place_container_plate demo_clean 50
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


POLICY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = POLICY_ROOT.parent.parent


def read_matrix(group, key, frame_id, valid_shapes):
    """Read one calibration matrix, including datasets stored without a time axis."""
    if key not in group:
        raise ValueError(f"Missing HDF5 dataset: {group.name}/{key}")
    dataset = group[key]
    if dataset.ndim == 2 and dataset.shape in valid_shapes:
        matrix = dataset[()]
    elif dataset.ndim == 3 and dataset.shape[1:] in valid_shapes:
        if not 0 <= frame_id < dataset.shape[0]:
            raise ValueError(
                f"Frame {frame_id} is outside {dataset.name} "
                f"with {dataset.shape[0]} frames."
            )
        matrix = dataset[frame_id]
    else:
        raise ValueError(
            f"Unexpected shape {dataset.shape} for {dataset.name}; "
            f"expected one of {valid_shapes}, optionally with a leading time axis."
        )
    matrix = np.asarray(matrix, dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Non-finite camera parameters in {dataset.name}.")
    return matrix


def extract_cam_params(episode_path, output_dir, camera="head_camera", frame_id=0):
    """Save one frame's CV extrinsics and [fx, fy, cx, cy], both as float32."""
    if frame_id < 0:
        raise ValueError("frame_id must be non-negative.")
    with h5py.File(episode_path, "r") as episode:
        camera_path = f"observation/{camera}"
        if camera_path not in episode:
            raise ValueError(f"Missing HDF5 camera group: {camera_path}")
        group = episode[camera_path]
        intrinsic = read_matrix(group, "intrinsic_cv", frame_id, ((3, 3),))
        extrinsic = read_matrix(group, "extrinsic_cv", frame_id, ((3, 4), (4, 4)))

    # Preserve the CV transform direction; only add the homogeneous bottom row.
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:extrinsic.shape[0], :] = extrinsic
    if not np.allclose(extrinsics[3], [0, 0, 0, 1]):
        raise ValueError("The extrinsic matrix must have bottom row [0, 0, 0, 1].")

    intrinsics = np.array(
        [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
        dtype=np.float32,
    )
    if np.any(intrinsics[:2] <= 0):
        raise ValueError("Camera focal lengths fx and fy must be positive.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extrinsics_path = output_dir / "camera_extrinsics.npy"
    intrinsics_path = output_dir / "camera_params.npy"
    np.save(extrinsics_path, extrinsics, allow_pickle=False)
    np.save(intrinsics_path, intrinsics, allow_pickle=False)
    return extrinsics_path, intrinsics_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_name", help="RoboTwin task name, e.g. click_bell.")
    parser.add_argument("task_config", nargs="?", default="demo_clean")
    parser.add_argument("expert_data_num", nargs="?", type=int, default=50,
                        help="Episode count used in the output directory name (default: 50).")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--frame-id", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data",
                        help="Raw RoboTwin data root (default: <repo>/data).")
    parser.add_argument("--training-data-root", type=Path,
                        default=POLICY_ROOT / "training_data",
                        help="Training data root (default: <policy>/training_data).")
    args = parser.parse_args()
    if args.expert_data_num <= 0:
        parser.error("expert_data_num must be positive.")
    if args.episode_id < 0 or args.frame_id < 0:
        parser.error("--episode-id and --frame-id must be non-negative.")

    episode_path = (
        args.data_root.expanduser().resolve() / args.task_name / args.task_config
        / "data" / f"episode{args.episode_id}.hdf5"
    )
    if not episode_path.is_file():
        parser.error(f"Source episode not found: {episode_path}")
    output_dir = (
        args.training_data_root.expanduser().resolve()
        / f"{args.task_name}-{args.task_config}-{args.expert_data_num}" / "cam_params"
    )
    try:
        paths = extract_cam_params(episode_path, output_dir, args.camera, args.frame_id)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Source: {episode_path}")
    print(f"Camera: {args.camera}, frame: {args.frame_id}")
    for path in paths:
        values = np.load(path, allow_pickle=False)
        print(f"Saved {path} (shape={values.shape}, dtype={values.dtype})")
        print(values)


if __name__ == "__main__":
    main()
