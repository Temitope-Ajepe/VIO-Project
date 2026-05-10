"""
Visualize feature matching between two consecutive frames
"""
import cv2
import numpy as np
import sys
sys.path.append('.')

from src.utils.data_loader import TUMVILoader
from src.vo.feature_detection import detect_features
from src.vo.feature_matching import match_features

np.random.seed(0)

# Load dataset
loader = TUMVILoader('data/dataset-room2_512_16')

# Load two consecutive frames
frame1 = loader.get_image(0)
frame2 = loader.get_image(1)

# Detect features
kp1, des1 = detect_features(frame1)
kp2, des2 = detect_features(frame2)

# Match features
matches = match_features(des1, des2)

print(f"Keypoints in frame 1: {len(kp1)}")
print(f"Keypoints in frame 2: {len(kp2)}")
print(f"Good matches found:   {len(matches)}")

# Draw matches
img_matches = cv2.drawMatches(
    frame1, kp1,
    frame2, kp2,
    matches[:50],  # show top 50 matches
    None,
    matchColor=(0, 255, 0),
    singlePointColor=(255, 0, 0),
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Save and show
cv2.imwrite('results/plots/feature_matching_room2.png', img_matches)
print("Saved to results/plots/feature_matching_room2.png")

cv2.imshow('Feature Matching - Room2', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()