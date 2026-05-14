"""
Feature Tracking using Optical Flow
Step 4 of the VO pipeline.
"""
import cv2
import numpy as np

np.random.seed(0)

LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def track_features(prev_img, curr_img, prev_pts):
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_img, curr_img, prev_pts, None, **LK_PARAMS
    )
    status = status.flatten()
    good_prev = prev_pts[status == 1]
    good_curr = curr_pts[status == 1]
    return good_prev, good_curr

def update_trajectory(trajectory, poses, R, t):
    """
    Update camera trajectory using proper pose composition.
    Stores full 4x4 pose matrix including rotation.
    """
    if len(poses) == 0:
        # First pose is identity
        pose = np.eye(4)
    else:
        # Build relative pose from R and t
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3]  = t.flatten()
        
        # Compose with previous pose
        pose = poses[-1] @ T_rel

    poses.append(pose.copy())
    trajectory.append(pose.copy())

    return trajectory, poses

def rotation_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion (qx, qy, qz, qw).
    """
    trace = R[0,0] + R[1,1] + R[2,2]

    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s
        qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s
        qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s
        qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s
        qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s
        qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s
        qz = 0.25 * s

    # Normalize quaternion
    norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    return qx/norm, qy/norm, qz/norm, qw/norm

def save_trajectory(trajectory, output_path, timestamps=None):
    """
    Save trajectory in proper TUM format:
    timestamp tx ty tz qx qy qz qw
    """


    with open(output_path, 'w') as f:
        for i, pose in enumerate(trajectory):
            ts = timestamps[i] if timestamps else float(i)

            if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
                tx, ty, tz = pose[:3, 3]
                R = pose[:3, :3]
                qx, qy, qz, qw = rotation_to_quaternion(R)
            else:
                tx, ty, tz = pose
                qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0

            f.write(
                f"{ts:.6f} {tx:.6f} {ty:.6f} {tz:.6f} "
                f"{qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n"
            )