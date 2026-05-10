"""
Trajectory Visualization
Plots estimated trajectory vs ground truth
for visual comparison and paper figures.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(0)

def plot_trajectory_2d(traj_est, traj_gt, 
                        sequence_name, save_path=None):
    """
    Plot estimated vs ground truth trajectory in 2D (top view).
    
    Args:
        traj_est: Nx3 estimated positions
        traj_gt:  Nx3 ground truth positions
        sequence_name: e.g. 'room2'
        save_path: where to save the plot (optional)
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ground truth
    plt.plot(
        traj_gt[:, 0], traj_gt[:, 2],
        color='red',
        linewidth=2,
        label='Ground Truth'
    )
    
    # Plot estimated trajectory
    plt.plot(
        traj_est[:, 0], traj_est[:, 2],
        color='blue',
        linewidth=2,
        linestyle='--',
        label='Estimated (VO)'
    )
    
    # Mark start and end points
    plt.scatter(traj_gt[0, 0], traj_gt[0, 2],
                color='green', s=100,
                zorder=5, label='Start')
    plt.scatter(traj_gt[-1, 0], traj_gt[-1, 2],
                color='black', s=100,
                zorder=5, label='End')
    
    plt.title(f'Trajectory Comparison — {sequence_name}')
    plt.xlabel('X (meters)')
    plt.ylabel('Z (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_trajectory_3d(traj_est, traj_gt,
                        sequence_name, save_path=None):
    """
    Plot estimated vs ground truth trajectory in 3D.
    
    Args:
        traj_est: Nx3 estimated positions
        traj_gt:  Nx3 ground truth positions
        sequence_name: e.g. 'room2'
        save_path: where to save the plot (optional)
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth
    ax.plot(
        traj_gt[:, 0], traj_gt[:, 1], traj_gt[:, 2],
        color='red', linewidth=2, label='Ground Truth'
    )
    
    # Plot estimated
    ax.plot(
        traj_est[:, 0], traj_est[:, 1], traj_est[:, 2],
        color='blue', linewidth=2,
        linestyle='--', label='Estimated (VO)'
    )
    
    ax.set_title(f'3D Trajectory — {sequence_name}')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"3D plot saved to {save_path}")
    
    plt.show()

def save_results_table(results, save_path):
    """
    Save ATE/RPE results as a CSV table.
    
    Args:
        results: dict of {sequence: {ate: x, rpe: y}}
        save_path: where to save the .csv file
    """
    import csv
    
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Sequence', 'Method', 
                         'ATE (m)', 'RPE (m/100m)'])
        
        for sequence, metrics in results.items():
            writer.writerow([
                sequence,
                metrics.get('method', 'VO'),
                f"{metrics['ate']:.4f}",
                f"{metrics['rpe']:.4f}"
            ])
    
    print(f"Results table saved to {save_path}")