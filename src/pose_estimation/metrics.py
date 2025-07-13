"""
metrics.py

Compute post-hoc analytics from session_data.npz:
- balance score (% frames off-balance)
- stroke segmentation via wrist-speed peaks
- elbow ROM per stroke + consistency stats
"""

import numpy as np

def compute_balance_score(lean_series, threshold=10):
    arr = np.array(lean_series, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum()==0: return np.nan
    return np.sum(np.abs(arr[valid])>threshold) / valid.sum() * 100.0

def segment_strokes(wrist_positions, fps,
                    vel_thresh=1000000,    # higher threshold
                    min_gap_s=0.8      # seconds between strokes
                   ):
    """
    wrist_positions: (N,2) array of x,y per frame
    fps: frames per second
    vel_thresh: speed threshold (px/s) to count a stroke
    min_gap_s: minimum time between successive strokes (in seconds)
    """
    w = np.array(wrist_positions, dtype=float)
    if w.shape[0] < 3:
        return []

    # 1) instantaneous speed
    dx = np.diff(w[:,0])
    dy = np.diff(w[:,1])
    speed = np.sqrt(dx*dx + dy*dy) * fps  # px/s

    peaks = []
    last_frame = -np.inf
    min_gap_frames = int(min_gap_s * fps)

    for i in range(1, len(speed)-1):
        # local maximum above threshold and outside refractory window
        if (speed[i] > vel_thresh
            and speed[i] > speed[i-1]
            and speed[i] > speed[i+1]
            and (i - last_frame) >= min_gap_frames):
            peaks.append(i+1)       # +1 to align with wrist_positions
            last_frame = i

    return peaks


def compute_elbow_rom(elbow_angles, stroke_indices, window=5):
    """
    elbow_angles: (N,) series
    stroke_indices: list of peak frames
    returns (rom_list, mean_rom, std_rom)
    """
    angles = np.array(elbow_angles, dtype=float)
    roms = []
    n = len(angles)
    for idx in stroke_indices:
        start = max(0, idx-window)
        end   = min(n, idx+window)
        seg = angles[start:end]
        seg = seg[~np.isnan(seg)]
        if seg.size>0:
            roms.append(seg.max() - seg.min())
    if not roms:
        return [], np.nan, np.nan
    return roms, np.mean(roms), np.std(roms)
