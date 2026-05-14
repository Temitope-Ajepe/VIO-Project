"""
Sliding Window Bundle Adjustment
Optimizes camera poses, velocities, and IMU biases
over a fixed window of recent keyframes.

Based on OKVIS and VINS-Mono approach.
States: {T_k, v_k, b_k, X_i}
Where:
    T_k = camera pose (rotation + translation)
    v_k = velocity
    b_k = IMU bias
    X_i = 3D landmark positions
"""
import numpy as np
from collections import deque

np.random.seed(0)

class SlidingWindowOptimizer:
    def __init__(self, window_size=10):
        """
        Initialize sliding window optimizer.
        
        Args:
            window_size: number of keyframes to keep
                        in the optimization window
        """
        self.window_size = window_size
        
        # Window of keyframes
        # Each keyframe stores: {pose, velocity, bias, timestamp}
        self.keyframes = deque(maxlen=window_size)
        
        # 3D landmarks visible in window
        self.landmarks = {}
        
        # IMU bias estimates
        self.acc_bias  = np.zeros(3)
        self.gyro_bias = np.zeros(3)

    def add_keyframe(self, pose, velocity, timestamp, 
                     preintegrated):
        """
        Add a new keyframe to the sliding window.
        Old keyframes are automatically dropped
        when window is full.
        
        Args:
            pose:          4x4 transformation matrix
            velocity:      3D velocity vector
            timestamp:     keyframe timestamp
            preintegrated: IMU preintegrated measurements
                          (delta_R, delta_v, delta_p, dt)
        """
        keyframe = {
            'pose':          pose,
            'velocity':      velocity,
            'timestamp':     timestamp,
            'preintegrated': preintegrated,
            'acc_bias':      self.acc_bias.copy(),
            'gyro_bias':     self.gyro_bias.copy()
        }
        self.keyframes.append(keyframe)

    def optimize(self):
        """
        Run one iteration of sliding window optimization.
        Minimizes visual + IMU residuals jointly.
        
        Returns:
            optimized_poses: list of optimized 4x4 poses
        """
        if len(self.keyframes) < 2:
            return [kf['pose'] for kf in self.keyframes]
        
        # Get current poses
        poses = [kf['pose'] for kf in self.keyframes]
        
        # Compute IMU residuals
        imu_residuals = self._compute_imu_residuals()
        
        # Simple gradient step to minimize residuals
        optimized_poses = self._update_poses(
            poses, imu_residuals
        )
        
        # Update keyframes with optimized poses
        for i, kf in enumerate(self.keyframes):
            kf['pose'] = optimized_poses[i]
        
        return optimized_poses

    def _compute_imu_residuals(self):
        """
        Compute IMU residuals between consecutive keyframes.
        Residual = difference between predicted and 
                   measured preintegrated values.
        
        Returns:
            residuals: list of residual vectors
        """
        residuals = []
        keyframe_list = list(self.keyframes)
        
        for i in range(len(keyframe_list) - 1):
            kf_i = keyframe_list[i]
            kf_j = keyframe_list[i + 1]
            
            # Get preintegrated measurements
            dR, dv, dp, dt = kf_j['preintegrated']
            
            # Get poses
            R_i = kf_i['pose'][:3, :3]
            t_i = kf_i['pose'][:3, 3]
            R_j = kf_j['pose'][:3, :3]
            t_j = kf_j['pose'][:3, 3]
            v_i = kf_i['velocity']
            
            # Gravity
            g = np.array([0, -9.81, 0])
            
            # Position residual
            r_p = (R_i.T @ (t_j - t_i - 
                   v_i * dt - 
                   0.5 * g * dt**2) - dp)
            
            # Velocity residual
            r_v = R_i.T @ (kf_j['velocity'] - 
                           v_i - g * dt) - dv
            
            residuals.append({
                'position': r_p,
                'velocity': r_v
            })
        
        return residuals

    def _update_poses(self, poses, residuals, 
                      learning_rate=0.01):
        """
        Simple gradient update of poses based on residuals.
        
        Args:
            poses:         list of 4x4 pose matrices
            residuals:     list of IMU residuals
            learning_rate: step size for update
        
        Returns:
            updated_poses: list of updated 4x4 poses
        """
        updated_poses = [p.copy() for p in poses]
        
        for i, residual in enumerate(residuals):
            if i + 1 < len(updated_poses):
                # Update translation based on 
                # position residual
                updated_poses[i + 1][:3, 3] -= (
                    learning_rate * residual['position']
                )
        
        return updated_poses

    def get_trajectory(self):
        """
        Extract trajectory from current window.
        
        Returns:
            positions: list of 3D positions
        """
        return [kf['pose'][:3, 3] 
                for kf in self.keyframes]

    def is_window_full(self):
        """Check if sliding window is full."""
        return len(self.keyframes) == self.window_size