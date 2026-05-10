"""
Essential Matrix Estimation
Step 3 of the VO pipeline.
Estimates camera rotation R and translation t
between two frames.
"""
import cv2
import numpy as np

np.random.seed(0)

def estimate_essential_matrix(kp1, kp2, matches, K):
    """
    Estimate Essential Matrix using matched keypoints.
    
    Args:
        kp1: keypoints from frame 1
        kp2: keypoints from frame 2
        matches: good matches from feature_matching.py
        K: camera intrinsic matrix (from config)
    
    Returns:
        E: Essential matrix
        mask: inlier mask from RANSAC
    """
    # Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # Estimate Essential Matrix with RANSAC
    # RANSAC filters out wrong matches automatically
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )
    
    return E, mask, pts1, pts2

def decompose_essential_matrix(E, pts1, pts2, K):
    """
    Decompose Essential Matrix into R and t.
    
    Args:
        E: Essential matrix
        pts1: matched points in frame 1
        pts2: matched points in frame 2
        K: camera intrinsic matrix
    
    Returns:
        R: rotation matrix (3x3)
        t: translation vector (3x1)
    """
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def get_camera_matrix(config):
    """
    Build camera intrinsic matrix K from config file.
    
    Args:
        config: loaded tum_vi.yaml config
    
    Returns:
        K: 3x3 intrinsic matrix
    """
    fx = config['camera']['fx']
    fy = config['camera']['fy']
    cx = config['camera']['cx']
    cy = config['camera']['cy']
    
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K