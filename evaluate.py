import numpy as np
import sys
sys.path.append('.')
from src.utils.visualization import plot_trajectory_2d
from src.evaluation.rpe import compute_rpe, print_results
from src.evaluation.ate import load_trajectory, compute_ate, align_trajectories, align_se3

# Load estimated trajectory
#traj_est = load_trajectory('results/trajectories/room2_vo.txt')
#traj_est = load_trajectory('results/trajectories/corridor3_vo.txt')
#traj_est = load_trajectory('results/trajectories/outdoors5_vo.txt')
#traj_est = load_trajectory('results/trajectories/room2_vio.txt')
traj_est = load_trajectory('results/trajectories/corridor3_vio.txt')

# Load ground truth
#gt_data = np.loadtxt('data/dataset-room2_512_16/dso/gt_imu.csv',
#                      delimiter=',', skiprows=1)
#gt_data = np.loadtxt('data/dataset-corridor3_512_16/dso/gt_imu.csv',
#                      delimiter=',', skiprows=1)
#gt_data = np.loadtxt('data/dataset-outdoors5_512_16/dso/gt_imu.csv',
#                      delimiter=',', skiprows=1)
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
# For VO use Sim(3):
# traj_est_aligned = align_trajectories(traj_est, traj_gt)

# For VIO use SE(3):
traj_est_aligned = align_se3(traj_est, traj_gt)

# Sim(3) alignment for visual comparison only
traj_est_sim3 = align_trajectories(traj_est, traj_gt)

# Compute ATE (after alignment)
ate = compute_ate(traj_est, traj_gt)
print(f'ATE: {ate:.4f} meters')

# Compute RPE
rpe = compute_rpe(traj_est_aligned, traj_gt)
print(f'RPE: {rpe:.4f} meters/100m')

# Print clean results table
#print_results('Room2', ate, rpe)
#print_results('Corridor3', ate, rpe)
#print_results('Outdoors5', ate, rpe)
#print_results('Room2 VIO', ate, rpe)
#print_results('Corridor3 VIO', ate, rpe)
print_results('Outdoors5 VIO', ate, rpe)


# Plot aligned trajectory
#plot_trajectory_2d(
#    traj_est_aligned, traj_gt,
#    'Room2', 'results/plots/room2_vo.png'
#)

#plot_trajectory_2d(
#    traj_est_aligned, traj_gt,
#    'Corridor3', 'results/plots/corridor3_vo.png'
#)

#plot_trajectory_2d(
#    traj_est_aligned, traj_gt,
#    'Outdoors5', 'results/plots/outdoors5_vo.png'
#)


# Plot 1: SE(3) — honest evaluation
#plot_trajectory_2d(traj_est_aligned, traj_gt,
#    'Room2 VIO SE3', 'results/plots/room2_vio_se3.png')

# Plot 2: Sim(3) — visual comparison
#plot_trajectory_2d(traj_est_sim3, traj_gt,
#    'Room2 VIO (Sim3 visual)', 'results/plots/room2_vio_sim3.png')

# SE(3) plot
#plot_trajectory_2d(traj_est_aligned, traj_gt,
#    'Corridor3 VIO SE3', 'results/plots/corridor3_vio_se3.png')

# Sim(3) plot
#plot_trajectory_2d(traj_est_sim3, traj_gt,
#    'Corridor3 VIO Sim3', 'results/plots/corridor3_vio_sim3.png')

# SE(3) plot
plot_trajectory_2d(traj_est_aligned, traj_gt,
    'Outdoors5 VIO SE3', 'results/plots/outdoors5_vio_se3.png')

# Sim(3) plot
plot_trajectory_2d(traj_est_sim3, traj_gt,
    'Outdoors5 VIO Sim3', 'results/plots/outdoors5_vio_sim3.png')