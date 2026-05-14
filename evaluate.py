import numpy as np
import sys
sys.path.append('.')
from src.utils.visualization import plot_trajectory_2d
from src.evaluation.ate import load_trajectory, compute_ate, align_trajectories
from src.evaluation.rpe import compute_rpe, print_results

# Load estimated trajectory
#traj_est = load_trajectory('results/trajectories/room2_vo.txt')
#traj_est = load_trajectory('results/trajectories/corridor3_vo.txt')
traj_est = load_trajectory('results/trajectories/outdoors5_vo.txt')
#traj_est = load_trajectory('results/trajectories/room2_vio.txt')

# Load ground truth
#gt_data = np.loadtxt('data/dataset-room2_512_16/dso/gt_imu.csv',
#                      delimiter=',', skiprows=1)
#gt_data = np.loadtxt('data/dataset-corridor3_512_16/dso/gt_imu.csv',
#                      delimiter=',', skiprows=1)
gt_data = np.loadtxt('data/dataset-outdoors5_512_16/dso/gt_imu.csv',
                      delimiter=',', skiprows=1)

traj_gt = gt_data[:, 1:4]


# Match lengths
min_len = min(len(traj_est), len(traj_gt))
traj_est = traj_est[:min_len]
traj_gt = traj_gt[:min_len]

# Align trajectory properly using Umeyama
traj_est_aligned = align_trajectories(traj_est, traj_gt)

# Compute ATE (after alignment)
ate = compute_ate(traj_est, traj_gt)
print(f'ATE: {ate:.4f} meters')

# Compute RPE
rpe = compute_rpe(traj_est_aligned, traj_gt)
print(f'RPE: {rpe:.4f} meters/100m')

# Print clean results table
#print_results('Room2', ate, rpe)
#print_results('Corridor3', ate, rpe)
print_results('Outdoors5', ate, rpe)
#print_results('Room2 VIO', ate, rpe)

# Plot aligned trajectory
#plot_trajectory_2d(
#    traj_est_aligned, traj_gt,
#    'Room2', 'results/plots/room2_vo.png'
#)

#plot_trajectory_2d(
#    traj_est_aligned, traj_gt,
#    'Corridor3', 'results/plots/corridor3_vo.png'
#)

plot_trajectory_2d(
    traj_est_aligned, traj_gt,
    'Outdoors5', 'results/plots/outdoors5_vo.png'
)
#plot_trajectory_2d(traj_est_aligned, traj_gt,
#    'Room2 VIO', 'results/plots/room2_vio.png')