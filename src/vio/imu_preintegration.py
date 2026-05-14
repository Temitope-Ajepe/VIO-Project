"""
IMU Preintegration
Integrates IMU measurements between keyframes
to estimate relative motion (delta R, delta v, delta p).

Based on: Forster et al. "IMU Preintegration on Manifold"
RSS 2015.
"""
import numpy as np

np.random.seed(0)

class IMUPreintegrator:
    def __init__(self, config):
        """
        Initialize IMU preintegrator with noise parameters.
        
        Args:
            config: loaded tum_vi.yaml config
        """
        # IMU noise parameters from config
        self.acc_noise  = config['imu']['accelerometer_noise']
        self.gyro_noise = config['imu']['gyroscope_noise']
        
        # Gravity vector (world frame, pointing down)
        self.gravity = np.array([0, -9.81, 0])
        
        # Reset preintegration state
        self.reset()

    def reset(self):
        """
        Reset preintegrated measurements.
        Called at the start of each keyframe interval.
        """
        # Delta rotation (3x3 matrix)
        self.delta_R = np.eye(3)
        
        # Delta velocity (3x1 vector)
        self.delta_v = np.zeros(3)
        
        # Delta position (3x1 vector)
        self.delta_p = np.zeros(3)
        
        # Time interval
        self.dt_sum = 0.0

    def integrate(self, gyro, acc, dt):
        """
        Integrate one IMU measurement.
        
        Args:
            gyro: gyroscope measurement (3,) in rad/s
                  measures angular velocity
            acc:  accelerometer measurement (3,) in m/s²
                  measures specific force
            dt:   time delta in seconds
        """
        # ── Step 1: Update rotation using gyroscope ──────
        # Convert angular velocity to rotation matrix
        omega = gyro  # angular velocity vector
        angle = np.linalg.norm(omega) * dt
        
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            # Rodrigues rotation formula
            K = self._skew_symmetric(axis)
            dR = (np.eye(3) + 
                  np.sin(angle) * K + 
                  (1 - np.cos(angle)) * K @ K)
        else:
            dR = np.eye(3)
        
        # ── Step 2: Update position using accelerometer ──
        # Remove gravity from accelerometer reading
        acc_world = self.delta_R @ acc
        
        # Update position (integrate twice)
        self.delta_p = (self.delta_p + 
                       self.delta_v * dt + 
                       0.5 * acc_world * dt**2)
        
        # Update velocity (integrate once)
        self.delta_v = self.delta_v + acc_world * dt
        
        # Update rotation
        self.delta_R = self.delta_R @ dR
        
        # Track total time
        self.dt_sum += dt

    def get_preintegrated(self):
        """
        Get preintegrated measurements between keyframes.
        
        Returns:
            delta_R: relative rotation (3x3)
            delta_v: relative velocity (3,)
            delta_p: relative position (3,)
            dt:      total time interval
        """
        return self.delta_R, self.delta_v, self.delta_p, self.dt_sum

    def _skew_symmetric(self, v):
        """
        Convert vector to skew-symmetric matrix.
        Used for Rodrigues rotation formula.
        
        Args:
            v: 3D vector
        Returns:
            3x3 skew-symmetric matrix
        """
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])


def load_imu_data(imu_path):
    """
    Load IMU data from TUM VI dataset.
    Format: timestamp, gx, gy, gz, ax, ay, az
    
    Args:
        imu_path: path to imu0/data.csv
    
    Returns:
        imu_data: Nx7 array
    """
    return np.loadtxt(imu_path, delimiter=',', skiprows=1)


def get_imu_between_frames(imu_data, t_start, t_end):
    """
    Extract IMU measurements between two timestamps.
    
    Args:
        imu_data: full IMU data array
        t_start:  start timestamp (seconds)
        t_end:    end timestamp (seconds)
    
    Returns:
        imu_segment: IMU measurements in time window
    """
    # Convert nanoseconds to seconds
    timestamps = imu_data[:, 0] * 1e-9
    
    mask = (timestamps >= t_start) & (timestamps <= t_end)
    return imu_data[mask]