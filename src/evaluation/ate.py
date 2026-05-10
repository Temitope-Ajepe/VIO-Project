"""
Absolute Trajectory Error (ATE)
Measures overall trajectory accuracy by comparing
estimated path vs ground truth path.
"""
import numpy as np

np.random.seed(0)

def align_trajectories(traj_est, traj_gt):
    """
    Align estimated trajectory to ground truth
    using Umeyama method (handles scale + rotation).
    
    Args:
        traj_est: Nx3 estimated positions
        traj_gt:  Nx3 ground truth positions
    
    Returns:
        traj_est_aligned: aligned estimated trajectory
    """
    # Center both trajectories
    mu_est = traj_est.mean(axis=0)
    mu_gt  = traj_gt.mean(axis=0)
    
    est_centered = traj_est - mu_est
    gt_centered  = traj_gt  - mu_gt
    
    # Compute scale
    scale = np.sqrt(
        (gt_centered ** 2).sum() / (est_centered ** 2).sum()
    )
    
    # Compute rotation using SVD
    W = gt_centered.T @ est_centered
    U, _, Vt = np.linalg.svd(W)
    R = U @ Vt
    
    # Apply alignment
    traj_est_aligned = scale * (est_centered @ R.T) + mu_gt
    
    return traj_est_aligned

def compute_ate(traj_est, traj_gt):
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        traj_est: Nx3 estimated positions
        traj_gt:  Nx3 ground truth positions
    
    Returns:
        ate: RMSE in meters (lower is better)
    """
    assert len(traj_est) == len(traj_gt), \
        "Trajectories must have same length!"
    
    # Align first
    traj_est_aligned = align_trajectories(traj_est, traj_gt)
    
    # Compute error
    errors = traj_est_aligned - traj_gt
    ate = np.sqrt(np.mean(np.sum(errors ** 2, axis=1)))
    
    return ate

def load_trajectory(filepath):
    """
    Load trajectory from TUM format .txt file.
    Format: timestamp tx ty tz qx qy qz qw
    
    Args:
        filepath: path to .txt trajectory file
    
    Returns:
        positions: Nx3 array of x, y, z positions
    """
    data = np.loadtxt(filepath)
    positions = data[:, 1:4]  # extract tx ty tz
    return positions