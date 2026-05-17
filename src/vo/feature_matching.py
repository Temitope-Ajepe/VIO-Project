"""
Feature Matching using BFMatcher
Step 2 of the VO pipeline.
Uses ratio test + cross-check as required.
"""
import cv2
import numpy as np

np.random.seed(0)

def match_features(des1, des2, ratio_threshold=0.75):
    """
    Match ORB descriptors between two frames
    using ratio test + cross-check.
    
    Args:
        des1: descriptors from frame 1
        des2: descriptors from frame 2
        ratio_threshold: Lowe's ratio test threshold
    
    Returns:
        good_matches: list of reliable matches
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Step 1: Ratio test (forward matching)
    matches_forward = bf.knnMatch(des1, des2, k=2)
    
    ratio_passed = set()
    for m, n in matches_forward:
        if m.distance < ratio_threshold * n.distance:
            ratio_passed.add(m.queryIdx)

    # Step 2: Cross-check (backward matching)
    matches_backward = bf.knnMatch(des2, des1, k=2)
    
    cross_check_passed = set()
    for m, n in matches_backward:
        if m.distance < ratio_threshold * n.distance:
            cross_check_passed.add(m.trainIdx)

    # Step 3: Keep only matches that passed both
    good_matches = []
    for m, n in matches_forward:
        if (m.queryIdx in ratio_passed and 
            m.queryIdx in cross_check_passed):
            good_matches.append(m)

    return good_matches

def visualize_matches(img1, kp1, img2, kp2, matches):
    """
    Draw matching features between two frames.
    """
    return cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )