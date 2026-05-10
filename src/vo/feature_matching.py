"""
Feature Matching using BFMatcher
Step 2 of the VO pipeline.
"""
import cv2
import numpy as np

np.random.seed(0)

def match_features(des1, des2, ratio_threshold=0.75):
    """
    Match ORB descriptors between two frames
    using ratio test to filter bad matches.
    
    Args:
        des1: descriptors from frame 1
        des2: descriptors from frame 2
        ratio_threshold: Lowe's ratio test threshold
    
    Returns:
        good_matches: list of reliable matches
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Find 2 nearest neighbours for each descriptor
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test - keep only good matches
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    
    return good_matches

def visualize_matches(img1, kp1, img2, kp2, matches):
    """
    Draw matching features between two frames.
    Good for debugging.
    """
    return cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )