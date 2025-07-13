#!/usr/bin/env python3
"""
demo.py

Load session_data.npz → compute and print key insights + plots for your boss.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from metrics import (
    compute_balance_score,
    segment_strokes,
    compute_elbow_rom
)

# load data
base = os.path.dirname(__file__)
data = np.load(os.path.join(base, 'session_data.npz'))
lean     = data['lean']
timestamps = data['timestamps']
elbow_L  = data['elbow_L']
wrist_R  = data['wrist_R']  # use right wrist for stroke detection

# 1) Balance Score
bal_score = compute_balance_score(lean, threshold=10)

# 2) Stroke segmentation
fps_est = 30  # adjust if known differently
strokes = segment_strokes(wrist_R, fps=fps_est, vel_thresh=50)

# 3) Elbow ROM stats
roms, mean_rom, std_rom = compute_elbow_rom(elbow_L, strokes, window=5)

# --- Print summary ---
print(f"\n=== TENNIS ANALYTICS SUMMARY ===")
print(f"Balance Score (>±10% lean): {bal_score:.1f}% of frames off-balance")
print(f"Detected strokes: {len(strokes)}")
print(f"Elbow ROM per stroke (°): {roms}")
print(f"→ Mean ROM: {mean_rom:.1f}°   StdDev: {std_rom:.1f}°\n")

# --- Plots for presentation ---
fig, axes = plt.subplots(3,1, figsize=(6,8))

# Lean over time
axes[0].plot(timestamps, lean, label='Lean (%)')
axes[0].axhline( 10, color='r', linestyle='--')
axes[0].axhline(-10, color='r', linestyle='--')
axes[0].set_title("Lean over Time")
axes[0].legend()

# Elbow angle & stroke peaks
axes[1].plot(elbow_L, label='Left Elbow Angle')
axes[1].scatter(strokes, elbow_L[strokes], color='r', label='Stroke Peaks')
axes[1].set_title("Elbow Angle & Stroke Timing")
axes[1].legend()

# ROM distribution
axes[2].hist(roms, bins=8, edgecolor='k')
axes[2].set_title("Elbow ROM Distribution")

plt.tight_layout()
plt.show()
