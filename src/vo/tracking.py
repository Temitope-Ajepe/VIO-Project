"""
Feature Tracking using Optical Flow
Step 4 of the VO pipeline.
Tracks features across consecutive frames
and builds the camera trajectory.
"""
import cv2
import numpy as np

np.random.seed(0)

# Optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def track_features(prev_img, curr_img, prev_pts):
    """
    Track features from previous frame to current frame
    using Lucas-Kanade Optical Flow.
    
    Args:
        prev_img: previous grayscale frame
        curr_img: current grayscale frame
        prev_pts: feature points from previous frame
    
    Returns:
        good_prev: successfully tracked points in prev frame
        good_curr: corresponding points in current frame
    """
    # Track points using optical flow
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_img, curr_img,
        prev_pts, None,
        **LK_PARAMS
    )
    
    # Keep only successfully tracked points
    status = status.flatten()
    good_prev = prev_img[status == 1] if prev_pts is not None else prev_pts
    good_curr = curr_pts[status == 1]
    good_prev = prev_pts[status == 1]
    
    return good_prev, good_curr

def update_trajectory(trajectory, poses, R, t):
    """
    Update camera trajectory using proper pose composition.
    
    Args:
        trajectory: list of previous camera positions
        poses: list of previous 4x4 transformation matrices
        R: rotation matrix
        t: translation vector
    
    Returns:
        updated trajectory, updated poses
    """
    if len(poses) == 0:
        # Start at identity (origin)
        pose = np.eye(4)
    else:
        # Compose new pose with previous
        prev_pose = poses[-1]
        new_pose = np.eye(4)
        new_pose[:3, :3] = R
        new_pose[:3, 3] = t.flatten()
        pose = prev_pose @ new_pose

    poses.append(pose)
    # Extract position from pose matrix
    position = pose[:3, 3]
    trajectory.append(position.copy())

    return trajectory, poses

def save_trajectory(trajectory, output_path):
    """
    Save trajectory in TUM format:
    timestamp tx ty tz qx qy qz qw
    
    Args:
        trajectory: list of camera positions
        output_path: where to save the .txt file
    """
    with open(output_path, 'w') as f:
        for i, pos in enumerate(trajectory):
            # Simplified - translation only for now
            f.write(f"{i} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                   f"0 0 0 1\n")