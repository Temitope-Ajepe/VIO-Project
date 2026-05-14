"""
Visual-Inertial Odometry Pipeline
Fuses camera and IMU data using:
- IMU Preintegration (Forster et al. 2015)
- Sliding Window Bundle Adjustment (OKVIS/VINS-Mono)
"""
import cv2
import numpy as np
import yaml
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.utils.data_loader import TUMVILoader
from src.vo.feature_detection import detect_features
from src.vo.feature_matching import match_features
from src.vo.essential_matrix import (estimate_essential_matrix,
                                      decompose_essential_matrix,
                                      get_camera_matrix)
from src.vo.tracking import update_trajectory, save_trajectory
from src.vio.imu_preintegration import (IMUPreintegrator,
                                         load_imu_data,
                                         get_imu_between_frames)
from src.vio.sliding_window import SlidingWindowOptimizer

np.random.seed(0)

def run_vio_pipeline(sequence_path, config_path, output_path):
    """
    Run the full VIO pipeline on a TUM VI sequence.
    
    Args:
        sequence_path: path to TUM VI sequence
        config_path:   path to tum_vi.yaml
        output_path:   where to save trajectory .txt
    """
    # ── 1. Load config and data ──────────────────────────
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    loader   = TUMVILoader(sequence_path)
    K        = get_camera_matrix(config)
    imu_path = os.path.join(
        sequence_path, 'mav0/imu0/data.csv'
    )
    imu_data = load_imu_data(imu_path)

    print(f"Loaded {len(loader)} frames")
    print(f"Loaded {len(imu_data)} IMU measurements")

    # ── 2. Initialize ────────────────────────────────────
    trajectory  = []
    poses       = []
    preint      = IMUPreintegrator(config)
    optimizer   = SlidingWindowOptimizer(window_size=10)

    prev_img       = loader.get_image(0)
    prev_kp, prev_des = detect_features(prev_img)
    prev_timestamp = loader.timestamps[0]

    # Initial pose at origin
    init_pose      = np.eye(4)
    init_velocity  = np.zeros(3)

    optimizer.add_keyframe(
        pose=init_pose,
        velocity=init_velocity,
        timestamp=prev_timestamp,
        preintegrated=(np.eye(3), 
                      np.zeros(3), 
                      np.zeros(3), 0.0)
    )

    # ── 3. Main loop ─────────────────────────────────────
    for i in range(1, len(loader)):
        curr_img       = loader.get_image(i)
        curr_timestamp = loader.timestamps[i]

        # Step 1: Detect and match features
        curr_kp, curr_des = detect_features(curr_img)
        matches = match_features(prev_des, curr_des)

        if len(matches) < 8:
            prev_img       = curr_img
            prev_kp, prev_des = curr_kp, curr_des
            prev_timestamp = curr_timestamp
            continue

        # Step 2: Estimate camera motion
        E, mask, pts1, pts2 = estimate_essential_matrix(
            prev_kp, curr_kp, matches, K
        )
        R, t = decompose_essential_matrix(E, pts1, pts2, K)

        # Step 3: Preintegrate IMU between frames
        preint.reset()
        imu_segment = get_imu_between_frames(
            imu_data, prev_timestamp, curr_timestamp
        )

        for j in range(len(imu_segment) - 1):
            gyro = imu_segment[j, 1:4]
            acc  = imu_segment[j, 4:7]
            dt   = ((imu_segment[j+1, 0] - 
                     imu_segment[j, 0]) * 1e-9)
            preint.integrate(gyro, acc, dt)

        preintegrated = preint.get_preintegrated()

        # Step 4: Build new pose from R and t
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3]  = t.flatten()

        if len(poses) > 0:
            new_pose = poses[-1] @ new_pose

        # Step 5: Add keyframe to sliding window
        optimizer.add_keyframe(
            pose=new_pose,
            velocity=preintegrated[1],
            timestamp=curr_timestamp,
            preintegrated=preintegrated
        )

        # Step 6: Optimize window
        optimized_poses = optimizer.optimize()

        # Step 7: Update trajectory
        position = optimized_poses[-1][:3, 3]
        trajectory.append(position.copy())
        poses.append(new_pose)

        if i % 50 == 0:
            print(f"Processed frame {i}/{len(loader)}")

        prev_img       = curr_img
        prev_kp, prev_des = curr_kp, curr_des
        prev_timestamp = curr_timestamp

    # ── 4. Save trajectory ───────────────────────────────
    save_trajectory(
    trajectory,
    output_path,
    timestamps=loader.timestamps[:len(trajectory)]
)
    print(f"VIO Trajectory saved to {output_path}")

    return trajectory


if __name__ == "__main__":
    os.makedirs("results/trajectories", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    run_vio_pipeline(
        sequence_path="data/dataset-room2_512_16",
        config_path="configs/tum_vi.yaml",
        output_path="results/trajectories/room2_vio.txt"
    )