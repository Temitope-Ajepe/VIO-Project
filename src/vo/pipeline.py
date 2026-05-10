"""
Monocular Visual Odometry Pipeline
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

np.random.seed(0)

def run_vo_pipeline(sequence_path, config_path, output_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    loader = TUMVILoader(sequence_path)
    K = get_camera_matrix(config)
    print(f"Loaded {len(loader)} frames from {sequence_path}")

    # Initialize
    trajectory = []
    poses = []
    prev_img = loader.get_image(0)
    prev_kp, prev_des = detect_features(prev_img)

    # Main loop
    for i in range(1, len(loader)):
        curr_img = loader.get_image(i)
        curr_kp, curr_des = detect_features(curr_img)
        matches = match_features(prev_des, curr_des)

        if len(matches) < 8:
            prev_img = curr_img
            prev_kp, prev_des = curr_kp, curr_des
            continue

        E, mask, pts1, pts2 = estimate_essential_matrix(
            prev_kp, curr_kp, matches, K
        )
        R, t = decompose_essential_matrix(E, pts1, pts2, K)
        trajectory, poses = update_trajectory(trajectory, poses, R, t)

        if i % 50 == 0:
            print(f"Processed frame {i}/{len(loader)}")

        prev_img = curr_img
        prev_kp, prev_des = curr_kp, curr_des

    save_trajectory(trajectory, output_path)
    print(f"Trajectory saved to {output_path}")
    return trajectory


""" 
if __name__ == "__main__":
    os.makedirs("results/trajectories", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    run_vo_pipeline(
        sequence_path="data/dataset-room2_512_16",
        config_path="configs/tum_vi.yaml",
        output_path="results/trajectories/room2_vo.txt"
    )#
    # """
""""
if __name__ == "__main__":
    os.makedirs("results/trajectories", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    run_vo_pipeline(
        sequence_path="data/dataset-corridor3_512_16",
        config_path="configs/tum_vi.yaml",
        output_path="results/trajectories/corridor3_vo.txt"
    )
"""


if __name__ == "__main__":
    os.makedirs("results/trajectories", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)


    run_vo_pipeline(
    sequence_path="data/dataset-outdoors5_512_16",
    config_path="configs/tum_vi.yaml",
    output_path="results/trajectories/outdoors5_vo.txt"
)