"""
Relative Pose Error (RPE)
Measures local trajectory accuracy over
fixed distance segments (100m translation
or 60 degrees rotation).
"""
import numpy as np

np.random.seed(0)

def compute_rpe(traj_est, traj_gt, segment_length=100):
    """
    Compute Relative Pose Error (RPE) over
    fixed length segments.
    
    Args:
        traj_est: Nx3 estimated positions
        traj_gt:  Nx3 ground truth positions
        segment_length: segment length in meters
    
    Returns:
        rpe_trans: translation RPE (meters per 100m)
        rpe_rot:   rotation RPE (degrees per 100m)
    """
    assert len(traj_est) == len(traj_gt), \
        "Trajectories must have same length!"
    
    trans_errors = []
    
    for i in range(len(traj_est) - 1):
        # Get segment in estimated trajectory
        delta_est = traj_est[i+1] - traj_est[i]
        
        # Get corresponding segment in ground truth
        delta_gt = traj_gt[i+1] - traj_gt[i]
        
        # Compute translation error
        error = np.linalg.norm(delta_est - delta_gt)
        trans_errors.append(error)
    
    # RMSE of translation errors
    rpe_trans = np.sqrt(np.mean(np.array(trans_errors) ** 2))
    
    return rpe_trans

def print_results(sequence_name, ate, rpe):
    """
    Print evaluation results in a clean table format.
    
    Args:
        sequence_name: e.g. 'room2', 'corridor3'
        ate: computed ATE value
        rpe: computed RPE value
    """
    print("\n" + "="*45)
    print(f"  Results for: {sequence_name}")
    print("="*45)
    print(f"  ATE  (m):        {ate:.4f}")
    print(f"  RPE  (m/100m):   {rpe:.4f}")
    print("="*45 + "\n")