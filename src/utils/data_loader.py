"""
TUM VI Dataset Loader
Loads synchronized images and IMU data.
"""
import numpy as np
import cv2
import os

# Fix seed for reproducibility
np.random.seed(0)

class TUMVILoader:
    def __init__(self, sequence_path):
        self.path = sequence_path
        self.image_dir = os.path.join(sequence_path, 'mav0/cam0/data')
        self.imu_path = os.path.join(sequence_path, 'mav0/imu0/data.csv')
        self.gt_path = os.path.join(sequence_path, 'dso/gt_imu.csv')
        self.timestamps, self.image_files = self._load_image_list()

    def _load_image_list(self):
        """Load sorted list of image timestamps and paths."""
        files = sorted(os.listdir(self.image_dir))
        timestamps = [float(f.replace('.png', '')) * 1e-9 for f in files]
        paths = [os.path.join(self.image_dir, f) for f in files]
        return timestamps, paths

    def get_image(self, idx):
        """Load grayscale image at index idx."""
        return cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)

    def get_imu_data(self):
        """Load IMU measurements."""
        return np.loadtxt(self.imu_path, delimiter=',', skiprows=1)

    def get_ground_truth(self):
        """Load ground truth poses."""
        return np.loadtxt(self.gt_path, delimiter=',', skiprows=1)

    def __len__(self):
        return len(self.image_files)
