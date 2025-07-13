#!/usr/bin/env python3
"""
media_pipe_pose.py

Core capture & visualization:
- BlazePose landmarks for tennis-relevant joints
- Color-coded skeleton + dashed guide lines at shoulder/hip centers (clipped to player bbox)
- Joint angles (elbow/knee) + smoothed “lean” metric
- Embedded live lean plot + on-screen playback controls
- Exports session_data.npz for metric post-processing
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse
from collections import deque
import matplotlib.pyplot as plt
import os

# -------------- Config & Pose Init --------------
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

JOINT_IDS = {
    'L_SHOULDER':11,'R_SHOULDER':12,
    'L_ELBOW':13,   'R_ELBOW':14,
    'L_WRIST':15,   'R_WRIST':16,
    'L_HIP':23,     'R_HIP':24,
    'L_KNEE':25,    'R_KNEE':26,
    'L_ANKLE':27,   'R_ANKLE':28,
}
LEFT_JOINTS  = {'L_SHOULDER','L_ELBOW','L_WRIST','L_HIP','L_KNEE','L_ANKLE'}
RIGHT_JOINTS = {'R_SHOULDER','R_ELBOW','R_WRIST','R_HIP','R_KNEE','R_ANKLE'}

CONNECTIONS = [
    ('L_SHOULDER','L_ELBOW'),('L_ELBOW','L_WRIST'),
    ('L_SHOULDER','L_HIP'),   ('L_HIP','L_KNEE'),('L_KNEE','L_ANKLE'),
    ('R_SHOULDER','R_ELBOW'),('R_ELBOW','R_WRIST'),
    ('R_SHOULDER','R_HIP'),   ('R_HIP','R_KNEE'),  ('R_KNEE','R_ANKLE'),
]

SMOOTH_ALPHA      = 0.3    # lean EMA α
LEAN_HISTORY_LEN  = 150
PLOT_UPDATE_EVERY = 5      # frames
OVERLAY_ALPHA     = 0.3
CONTROLS_TEXT     = "Controls: p=Pause   f=Faster   s=Slower   q=Quit"

# -------------- Helpers --------------
def calc_angle(a,b,c):
    va = np.array(a)-np.array(b)
    vc = np.array(c)-np.array(b)
    cos = np.dot(va,vc)/(np.linalg.norm(va)*np.linalg.norm(vc)+1e-6)
    return np.degrees(np.arccos(np.clip(cos,-1,1)))

def draw_dashed_line(img, p1, p2, color, thickness=2, dash_len=8):
    x1,y1 = p1; x2,y2 = p2
    dist = int(np.hypot(x2-x1, y2-y1))
    for i in range(0, dist, dash_len*2):
        s = i/dist; e = min((i+dash_len)/dist,1.0)
        xa = int(x1 + (x2-x1)*s); ya = int(y1 + (y2-y1)*s)
        xb = int(x1 + (x2-x1)*e); yb = int(y1 + (y2-y1)*e)
        cv2.line(img,(xa,ya),(xb,yb),color,thickness)

# -------------- CLI --------------
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='0',
                    help='video file or webcam idx')
args = parser.parse_args()
src = int(args.source) if args.source.isdigit() else args.source

# -------------- Setup Plot --------------
lean_history = deque(maxlen=LEAN_HISTORY_LEN)
lean_prev    = None

plt.ion()
fig, ax = plt.subplots(figsize=(2,1.5))
line, = ax.plot([],[],lw=2,label="Lean %")
ax.set_ylim(-50,50)
ax.set_xlim(0,LEAN_HISTORY_LEN)
ax.legend(loc="upper right")

# -------------- Export Buffers --------------
lean_series    = []
elbow_L_series = []
elbow_R_series = []
wrist_L_series = []
wrist_R_series = []
timestamps     = []

# -------------- Video Loop --------------
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000/fps)
speed = 1.0
frame_idx = 0
plot_img = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    timestamps.append(frame_idx/fps)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(rgb)
    overlay = frame.copy()
    pts = {}

    if res.pose_landmarks:
        # draw landmarks
        for name, idx in JOINT_IDS.items():
            lm = res.pose_landmarks.landmark[idx]
            x, y = int(lm.x*w), int(lm.y*h)
            pts[name] = (x, y)
            col = (255,0,0) if name in RIGHT_JOINTS else (0,0,255)
            cv2.circle(overlay, (x,y), 5, col, -1)

        # draw skeleton
        for a, b in CONNECTIONS:
            if a in pts and b in pts:
                col = (255,0,0) if a in RIGHT_JOINTS else (0,0,255)
                cv2.line(overlay, pts[a], pts[b], col, 2)

        # compute centers
        l_sh, r_sh = pts['L_SHOULDER'], pts['R_SHOULDER']
        l_hp, r_hp = pts['L_HIP'],      pts['R_HIP']
        sh_cx = (l_sh[0] + r_sh[0]) / 2
        hp_cx = (l_hp[0] + r_hp[0]) / 2
        hp_cy = (l_hp[1] + r_hp[1]) / 2

        # lean & smooth
        raw = 100 * (sh_cx - hp_cx) / w
        sm  = raw if lean_prev is None else SMOOTH_ALPHA*raw + (1-SMOOTH_ALPHA)*lean_prev
        lean_prev = sm
        lean_history.append(sm)

        # record exports
        lean_series.append(sm)
        # elbow angles
        a,b,c = pts['L_SHOULDER'], pts['L_ELBOW'], pts['L_WRIST']
        angL = calc_angle(a,b,c); elbow_L_series.append(angL)
        a,b,c = pts['R_SHOULDER'], pts['R_ELBOW'], pts['R_WRIST']
        angR = calc_angle(a,b,c); elbow_R_series.append(angR)
        # wrists
        wrist_L_series.append(pts['L_WRIST'])
        wrist_R_series.append(pts['R_WRIST'])

        # ─── Compute bbox for clipping ───
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # ─── Draw clipped guide lines ───
        guide_col = (0,255,255)
        draw_dashed_line(overlay,
                         (int(sh_cx), y_min),
                         (int(sh_cx), y_max),
                         guide_col)
        draw_dashed_line(overlay,
                         (x_min, int(hp_cy)),
                         (x_max, int(hp_cy)),
                         guide_col)

        # on-frame text
        cv2.putText(overlay, f"{int(angL)}°",
                    (pts['L_ELBOW'][0]+5, pts['L_ELBOW'][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(overlay, f"{int(angR)}°",
                    (pts['R_ELBOW'][0]+5, pts['R_ELBOW'][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(overlay, f"Lean: {sm:+.1f}%",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    else:
        # fill NaNs when no detection
        lean_series.append(lean_prev or 0)
        elbow_L_series.append(np.nan)
        elbow_R_series.append(np.nan)
        wrist_L_series.append((np.nan,np.nan))
        wrist_R_series.append((np.nan,np.nan))

    # blend & controls prompt
    frame = cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1-OVERLAY_ALPHA, 0)
    cv2.putText(frame, CONTROLS_TEXT, (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # update & embed plot
    if frame_idx % PLOT_UPDATE_EVERY == 0 and lean_history:
        data = list(lean_history)
        line.set_xdata(range(len(data)))
        line.set_ydata(data)
        ax.set_xlim(0, max(len(data), LEAN_HISTORY_LEN))
        fig.canvas.draw()
        w_fig, h_fig = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h_fig, w_fig, 4)
        plot_img = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        target_w = int(w * 0.25)
        scale = target_w / w_fig
        plot_img = cv2.resize(plot_img, None,
                              fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)

    if plot_img is not None:
        ph, pw = plot_img.shape[:2]
        frame[0:ph, w-pw:w] = plot_img

    cv2.imshow("Tennis Pose Estimation", frame)

    key = cv2.waitKey(int(delay / speed)) & 0xFF
    if   key == ord('q'): break
    elif key == ord('p'):
        while cv2.waitKey(0) & 0xFF != ord('p'):
            pass
    elif key == ord('f'):
        speed *= 1.2
    elif key == ord('s'):
        speed = max(0.1, speed / 1.2)

    frame_idx += 1

# -------------- Cleanup & Export --------------
cap.release()
cv2.destroyAllWindows()
pose.close()

out_path = os.path.join(os.path.dirname(__file__), 'session_data.npz')
np.savez(out_path,
    timestamps = np.array(timestamps),
    lean       = np.array(lean_series),
    elbow_L    = np.array(elbow_L_series),
    elbow_R    = np.array(elbow_R_series),
    wrist_L    = np.array(wrist_L_series),
    wrist_R    = np.array(wrist_R_series),
)
print(f"Session data saved to {out_path}")
