"""
Feature Detection using ORB
Step 1 of the VO pipeline.
"""
import cv2
import numpy as np

np.random.seed(0)

def detect_features(image, n_features=2000):
    """
    Detect ORB keypoints and descriptors.
    
    Args:
        image: grayscale image
        n_features: max number of features to detect
    
    Returns:
        keypoints: list of detected points
        descriptors: feature descriptions
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def visualize_keypoints(image, keypoints):
    """
    Draw keypoints on image for debugging.
    
    Args:
        image: grayscale image
        keypoints: detected keypoints
    
    Returns:
        image with keypoints drawn in green
    """
    return cv2.drawKeypoints(
        image, keypoints, None, color=(0, 255, 0)
    )
