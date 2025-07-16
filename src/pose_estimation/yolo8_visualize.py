#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def visualize_pose(
    video_path: str,
    model_name: str = "yolov8n-pose",
    output_path: str = None,
    display: bool = True,
    wait_ms: int = 1
):
    """
    Runs YOLOv8-Pose on `video_path`, overlays keypoints, and:
     - if display=True: shows live window
     - if output_path given: writes annotated .mp4 to that file
    """
    model = YOLO(model_name)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Stream infer + plot
    for result in model.track(source=video_path, stream=True, verbose=False):
        # result.plot() returns the frame with keypoints overlaid
        annotated = result.plot()  # np.ndarray, HxWx3

        if display:
            cv2.imshow("YOLOv8-Pose Tracking", annotated)
            if cv2.waitKey(wait_ms) == 27:  # ESC to quit early
                break

        if writer:
            # convert RGB->BGR if needed
            if annotated.shape[2] == 3:
                bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
            else:
                bgr = annotated
            writer.write(bgr)

    if display:
        cv2.destroyAllWindows()
    if writer:
        writer.release()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s","--source",      required=True,
                   help="path to input video")
    p.add_argument("-m","--model",       default="yolov8n-pose",
                   help="YOLOv8-Pose model (e.g. yolov8n-pose, yolov8s-pose)")
    p.add_argument("-o","--output",      default=None,
                   help="optional path to save annotated mp4")
    p.add_argument("-d","--no-display",  action="store_true",
                   help="don't open a live window")
    p.add_argument("-w","--wait_ms",     type=int, default=1,
                   help="ms to wait between frames in display")
    args = p.parse_args()

    visualize_pose(
        video_path=args.source,
        model_name=args.model,
        output_path=args.output,
        display=not args.no_display,
        wait_ms=args.wait_ms,
    )

if __name__ == "__main__":
    main()
