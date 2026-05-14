"""
Start-End Drift Evaluation
For Corridor3 and Outdoors5 sequences
where ground truth is only available
at start and end.
"""
import numpy as np
import sys
sys.path.append('.')
from src.evaluation.ate import load_trajectory

def compute_start_end_drift(traj_est, traj_gt):
    """
    Compute drift between estimated and ground truth
    end positions after aligning start positions.
    
    Args:
        traj_est: Nx3 estimated positions
        traj_gt:  Nx3 ground truth positions
    
    Returns:
        drift: end position error in meters
    """
    # Align start positions
    start_offset = traj_gt[0] - traj_est[0]
    traj_est_aligned = traj_est + start_offset

    # Compute end position error
    end_est = traj_est_aligned[-1]
    end_gt  = traj_gt[-1]
    drift   = np.linalg.norm(end_est - end_gt)

    return drift, traj_est_aligned

def print_drift_results(sequence, drift):
    print(f"\n Drift Results: {sequence}")
    print(f"Start-End Drift: {drift:.4f} meters")
    print(f"\n")


#Corridor3 
print("Computing drift for Corridor3")
traj_est = load_trajectory('results/trajectories/corridor3_vo.txt')
gt_data  = np.loadtxt(
    'data/dataset-corridor3_512_16/dso/gt_imu.csv',
    delimiter=',', skiprows=1
)
traj_gt  = gt_data[:, 1:4]
min_len  = min(len(traj_est), len(traj_gt))
traj_est = traj_est[:min_len]
traj_gt  = traj_gt[:min_len]

drift_corridor3, _ = compute_start_end_drift(traj_est, traj_gt)
print_drift_results('Corridor3', drift_corridor3)

#  Outdoors5 
print("Computing drift for Outdoors5...")
traj_est2 = load_trajectory('results/trajectories/outdoors5_vo.txt')
gt_data2  = np.loadtxt(
    'data/dataset-outdoors5_512_16/dso/gt_imu.csv',
    delimiter=',', skiprows=1
)
traj_gt2  = gt_data2[:, 1:4]
min_len2  = min(len(traj_est2), len(traj_gt2))
traj_est2 = traj_est2[:min_len2]
traj_gt2  = traj_gt2[:min_len2]

drift_outdoors5, _ = compute_start_end_drift(traj_est2, traj_gt2)
print_drift_results('Outdoors5', drift_outdoors5)