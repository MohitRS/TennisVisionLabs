#!/usr/bin/env python3
"""
yolo8_full_pipeline_for_coaches.py

– Frame-by-frame YOLOv8-Pose inference
– Robust filtering and dynamic thresholds
– Computes tennis metrics
– Outputs a coach-friendly summary
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO

def extract_keypoints(video_path, model_name, visualize=False, output_path=None):
    model = YOLO(model_name)
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    kps_list, ts_list = [], []
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8-Pose on this frame
        res = model(frame, verbose=False)[0]

        # Visualization if requested
        if visualize or writer:
            ann = res.plot()
            bgr = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
            if visualize:
                cv2.imshow('Pose', bgr)
                if cv2.waitKey(1) == 27:
                    break
            if writer:
                writer.write(bgr)

        # Extract 17 keypoints (x, y, conf) for first detected person
        frame_kps = np.zeros((17,3), dtype=float)
        kpt = res.keypoints
        if kpt is not None:
            try:
                arr = kpt.data.cpu().numpy()
            except:
                arr = kpt.numpy()
            # arr shape: (n_people,17,3) or (17,3)
            if arr.ndim == 3 and arr.shape[0] > 0:
                frame_kps = arr[0]
            elif arr.ndim == 2 and arr.shape == (17,3):
                frame_kps = arr

        kps_list.append(frame_kps)
        ts_list.append(idx / fps)
        idx += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    if writer:
        writer.release()

    return np.stack(kps_list), np.array(ts_list), fps

def compute_metrics(kps, ts, fps, args):
    N = kps.shape[0]
    # 1) Filter frames by overall confidence and critical joints
    conf_frame = kps[:,:,2].mean(axis=1)
    keep_conf  = conf_frame >= args.conf_thresh
    joints_idx = [5,6,15,16]  # shoulders and ankles
    keep_joints = np.all(kps[:, joints_idx, 2] >= args.joint_conf_thresh, axis=1)
    keep = keep_conf & keep_joints

    kps_f = kps[keep]
    ts_f  = ts[keep]
    Nf    = kps_f.shape[0]

    # 2) Balance (lean)
    ls = kps_f[:,5,:2]
    rs = kps_f[:,6,:2]
    mid_sh = (ls + rs) / 2
    ft = (kps_f[:,15,:2] + kps_f[:,16,:2]) / 2
    sw = np.linalg.norm(ls - rs, axis=1)
    valid_sw = sw >= args.min_shoulder_dist
    lean = ((mid_sh[:,0] - ft[:,0]) / (sw/2 + 1e-6)) * 100
    lean = lean[valid_sw]
    off_balance_pct = float((np.abs(lean) > args.bal_thresh).mean() * 100)

    # 3) Strokes & speed
    wrist = kps_f[:,10,:2]
    dx, dy = np.diff(wrist[:,0]), np.diff(wrist[:,1])
    speed_all = np.sqrt(dx*dx + dy*dy) * fps
    median_speed = float(np.median(speed_all))
    vel_thresh = max(args.vel_thresh, args.speed_factor * median_speed)

    # Detect peaks for strokes
    peaks = []
    last = -np.inf
    gap = int(args.gap_s * fps)
    for i in range(1, len(speed_all)-1):
        if (speed_all[i] > vel_thresh and
            speed_all[i] > speed_all[i-1] and speed_all[i] > speed_all[i+1] and
            (i - last) >= gap):
            peaks.append(i+1)
            last = i

    # 4) Elbow range-of-motion
    def angle_series(side):
        idxs = (5,7,9) if side=='left' else (6,8,10)
        sh = kps_f[:,idxs[0],:2]
        el = kps_f[:,idxs[1],:2]
        wr = kps_f[:,idxs[2],:2]
        v1, v2 = sh - el, wr - el
        cos = np.einsum('ij,ij->i', v1, v2) / (
              np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)+1e-6)
        return np.degrees(np.arccos(np.clip(cos, -1,1)))
    ang_r = angle_series('right')
    ang_l = angle_series('left')

    def compute_rom(angles):
        roms = []
        for idx in peaks:
            seg = angles[max(0,idx-args.window):min(len(angles),idx+args.window)]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                r = float(seg.max() - seg.min())
                if r <= args.max_rom:
                    roms.append(r)
        return roms
    rom_r = compute_rom(ang_r)
    rom_l = compute_rom(ang_l)

    # 5) Symmetry index
    mean_r = float(np.mean(rom_r)) if rom_r else np.nan
    mean_l = float(np.mean(rom_l)) if rom_l else np.nan
    sym_idx = float(abs(mean_r-mean_l)/((mean_r+mean_l)/2+1e-6)*100) if (mean_r+mean_l)>0 else np.nan

    # 6) Compile metrics
    return {
        'total_frames': N,
        'kept_frames': Nf,
        'off_balance_pct': off_balance_pct,
        'median_lean': float(np.median(lean)),
        'mean_lean': float(np.mean(lean)),
        'strokes': len(peaks),
        'median_stroke_speed': float(median_speed),
        'mean_stroke_speed': float(np.mean(speed_all)),
        'median_rom_r': float(np.median(rom_r)),
        'mean_rom_r': float(np.mean(rom_r)),
        'median_rom_l': float(np.median(rom_l)),
        'mean_rom_l': float(np.mean(rom_l)),
        'symmetry_index': sym_idx,
        'median_stance_width': float(np.median(sw[valid_sw])),
        'mean_stance_width': float(np.mean(sw[valid_sw])),
        'median_twist_velocity': float(np.median(
            np.gradient(
                np.degrees(np.arctan2(mid_sh[:,1] - ft[:,1], mid_sh[:,0] - ft[:,0])),
                1.0/fps
            )
        )),
    }

def print_coach_report(m):
    print("\n=== COACH-FRIENDLY REPORT ===")
    print(f"Stability: {m['off_balance_pct']:.1f}% off-balance frames (goal <20%).")
    print(f"Rally length: {m['strokes']} strokes detected.")
    print(f"Stroke speed: {m['mean_stroke_speed']:.1f}px/s avg (median {m['median_stroke_speed']:.1f}).")
    print(f"Elbow ROM: Right {m['median_rom_r']:.1f}° median, Left {m['median_rom_l']:.1f}° median.")
    print(f"Symmetry index: {m['symmetry_index']:.1f}% (goal <10%).")
    print(f"Stance width: {m['mean_stance_width']:.1f}px avg (median {m['median_stance_width']:.1f}).")
    print(f"Torso twist vel: {m['median_twist_velocity']:.1f}°/s median.\n")
    print("Recommendations:")
    print(" - Aim to reduce off-balance frames below 20%.")
    print(" - Consistency: maintain stable stroke speeds.")
    print(" - Symmetry: keep elbow ROM difference under 10%.")
    print(" - Stance: hold a consistent stance width for power & balance.\n")

def main():
    p = argparse.ArgumentParser(description="YOLOv8 Tennis Analytics for Coaches")
    p.add_argument("-s","--source",      required=True, help="input video")
    p.add_argument("-m","--model",       default="yolov8n-pose", help="YOLOv8-Pose model")
    p.add_argument("--conf_thresh",      type=float, default=0.5, help="min avg frame conf")
    p.add_argument("--joint_conf_thresh",type=float, default=0.5, help="min conf for shoulders/feet")
    p.add_argument("--min_shoulder_dist",type=float, default=40,  help="min px shoulder width")
    p.add_argument("--vel_thresh",       type=float, default=150, help="base px/s stroke thresh")
    p.add_argument("--speed_factor",     type=float, default=2.0, help="multiplier × median speed")
    p.add_argument("--bal_thresh",       type=float, default=10,  help="% lean threshold")
    p.add_argument("--gap_s",            type=float, default=0.8, help="sec between strokes")
    p.add_argument("--window",           type=int,   default=5,   help="frames around stroke")
    p.add_argument("--max_rom",          type=float, default=180, help="max physiological ROM")
    p.add_argument("--visualize",        action="store_true", help="show live overlay")
    p.add_argument("--output",           default=None, help="save annotated video path")
    args = p.parse_args()

    kps, ts, fps = extract_keypoints(args.source, args.model,
                                     visualize=args.visualize,
                                     output_path=args.output)
    metrics = compute_metrics(kps, ts, fps, args)
    print_coach_report(metrics)

if __name__ == '__main__':
    main()
