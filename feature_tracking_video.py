"""
Feature Tracking Video
Shows camera feed with tracked features
drawn on each frame.
"""
import cv2
import numpy as np
import sys
sys.path.append('.')

from src.utils.data_loader import TUMVILoader
from src.vo.feature_detection import detect_features
from src.vo.feature_matching import match_features

np.random.seed(0)

# ── Settings ─────────────────────────────────────────────
SEQUENCE    = 'data/dataset-room2_512_16'
OUTPUT_PATH = 'results/plots/feature_tracking_room2.avi'
MAX_FRAMES  = 300  # how many frames to process
# ─────────────────────────────────────────────────────────

# Load dataset
loader = TUMVILoader(SEQUENCE)
print(f"Loaded {len(loader)} frames")

# Video writer setup
frame_sample = loader.get_image(0)
h, w = frame_sample.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, 20.0, (w*2, h))

# Initialize
prev_img = loader.get_image(0)
prev_kp, prev_des = detect_features(prev_img)

print(f"Creating video... ({MAX_FRAMES} frames)")

for i in range(1, min(MAX_FRAMES, len(loader))):
    curr_img = loader.get_image(i)
    curr_kp, curr_des = detect_features(curr_img)

    # Match features
    matches = match_features(prev_des, curr_des)

    # Convert to BGR for color drawing
    prev_bgr = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)
    curr_bgr = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)

    # Draw matched features
    for m in matches[:50]:
        # Get point coordinates
        pt1 = tuple(map(int, prev_kp[m.queryIdx].pt))
        pt2 = tuple(map(int, curr_kp[m.trainIdx].pt))

        # Draw circle on current frame
        cv2.circle(curr_bgr, pt2, 4, (0, 255, 0), -1)

        # Draw circle on previous frame
        cv2.circle(prev_bgr, pt1, 4, (0, 0, 255), -1)

        # Draw line showing movement
        cv2.line(curr_bgr, pt1, pt2, (255, 0, 0), 1)

    # Add frame info text
    cv2.putText(curr_bgr,
                f'Frame {i}/{MAX_FRAMES} | Matches: {len(matches)}',
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    # Side by side: previous | current
    combined = np.hstack([prev_bgr, curr_bgr])
    out.write(combined)

    # Show live preview
    cv2.imshow('Feature Tracking', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if i % 50 == 0:
        print(f"Processed frame {i}/{MAX_FRAMES}")

    prev_img = curr_img
    prev_kp, prev_des = curr_kp, curr_des

# Cleanup
out.release()
cv2.destroyAllWindows()
print(f"Video saved to {OUTPUT_PATH}")