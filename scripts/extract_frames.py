import cv2
import os

# paths
video_path = r"data/raw/Rally1_junaidraw.mp4"
out_dir   = r"data/processed/player_detection/images"
os.makedirs(out_dir, exist_ok=True)

# capture
cap = cv2.VideoCapture(video_path)
count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # every 30th frame
    if count % 30 == 0:
        fn = f"frame_{saved:04d}.jpg"
        cv2.imwrite(os.path.join(out_dir, fn), frame)
        saved += 1
    count += 1

cap.release()
print(f"Extracted {saved} frames to {out_dir}")
