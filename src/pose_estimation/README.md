
# 🎾 TennisVisionLabs — Pose Estimation Module

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

> **Two complementary pipelines** for extracting tennis‐specific biomechanics from video:  
> – **Mediapipe (CPU-friendly)** for rapid prototyping  
> – **YOLOv8+Pose (GPU-accelerated)** for high-accuracy joint tracking  

---

## 🚀 Table of Contents

1. [Features](#-features)  
2. [Methodology](#-methodology)  
   1. [Data Ingestion](#data-ingestion)  
   2. [Frame Extraction](#frame-extraction)  
   3. [Pose Detection](#pose-detection)  
   4. [Rally & Stroke Segmentation](#rally--stroke-segmentation)  
   5. [Metric Computation](#metric-computation)  
   6. [Reporting & Visualization](#reporting--visualization)  
3. [Installation](#-installation)  
4. [Usage](#-usage)  
   - [Mediapipe Pipeline](#mediapipe-pipeline)  
   - [YOLOv8 Pipeline](#yolov8-pipeline)  
5. [Configuration](#-configuration)  
6. [Project Structure](#-project-structure)  
7. [Extending & Customizing](#-extending--customizing)  
8. [Contributing](#-contributing)  
9. [License](#-license)  

---

## ✨ Features

- 📊 **Stability & Balance Score**: % of off‐balance frames (>±10% lean)  
- 🎾 **Stroke Count & Rally Segmentation**  
- 💪 **Elbow ROM** (range of motion) per stroke (min/median/max)  
- ⚖️ **Symmetry Index**: left vs. right ROM imbalance  
- 📐 **Stance Width** & **Torso Rotation Velocity**  
- 🏷️ **Session Persistence**: raw joints + metrics in `session_data.npz`  
- 🎥 **Visual Overlays**: skeletons, bounding boxes, metrics HUD  

---

## 🧠 Methodology

### 1. Data Ingestion
- **Input**: raw match videos (`.mp4`, `.avi`) in `data/raw/`
- **Validation**: checks file readability & frame rate

### 2. Frame Extraction
- **Stride Control**: adjustable frame skipping (`--stride`)
- **Storage**: cached in `data/processed/frames/` for repeat runs

### 3. Pose Detection
| Pipeline          | Library         | Key File             | Device      |
|-------------------|-----------------|----------------------|-------------|
| **Mediapipe**     | `mediapipe`     | `media_pipe_pose.py` | CPU         |
| **Ultralytics YOLOv8+Pose** | `ultralytics` | `yolo8_pipeline.py`  | GPU/CPU     |

- **Mediapipe**  
  - Uses BlazePose  
  - Configurable complexity & smoothing  
- **YOLOv8+Pose**  
  - Uses Ultralytics pose weights (`yolov8n-pose.pt`)  
  - Bounding-box → keypoint regression  

### 4. Rally & Stroke Segmentation
- **Lean Thresholding**: detect off-balance transitions  
- **Motion Peaks**: identify stroke events via wrist/elbow velocity  
- **Temporal Clustering**: group frames into rallies  

### 5. Metric Computation
- Implemented in `metrics.py`  
- Input: per-frame landmarks → pandas `DataFrame`  
- Output: dict of KPIs (stability, ROM, symmetry, stance, twist)  

### 6. Reporting & Visualization
- **Text Report**: `report.txt` with summary table  
- **Visualized Video**: skeleton + KPIs overlay (`--visualize`)  
- **Session File**: `session_data.npz` for downstream analysis  

---

## ⚙️ Installation

```bash
# 1. Clone repo
git clone https://github.com/MohitRS/TennisVisionLabs.git
cd TennisVisionLabs

# 2. Create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
````

---

## 🏃 Usage

### Mediapipe Pipeline

```bash
python src/pose_estimation/media_pipe_pose.py \
  --source data/raw/Rally1_junaidraw.mp4 \
  --stride 1 \
  --visualize
```

* **--stride**: process every \_n\_th frame (default: 1)
* **--visualize**: output `visualized.mp4` in `runs/mp_pose/...`

### YOLOv8 Pipeline

```bash
python src/pose_estimation/yolo8_pipeline.py \
  --source data/raw/Rally1_junaidraw.mp4 \
  --weights yolov8n-pose.pt \
  --conf 0.4 \
  --visualize \
  --save-csv
```

* **--weights**: path to Ultralytics pose weights
* **--conf**: detection confidence threshold
* **--save-csv**: export per-frame landmarks to CSV

---

## 🔧 Configuration

Edit top-of-file constants:

* **Mediapipe**: `MODEL_COMPLEXITY`, smoothing
* **YOLOv8**: default `CONF`, `DEVICE` (CPU/GPU)
* **Metrics**: thresholds in `metrics.py`

---

## 📁 Project Structure

```
TennisVisionLabs/
├─ data/
│  ├─ raw/            # Unprocessed videos
│  └─ processed/      # Cached frames & artifacts
├─ src/pose_estimation/
│  ├─ media_pipe_pose.py
│  ├─ yolo8_pipeline.py
│  ├─ yolo8_visualize.py
│  ├─ metrics.py
│  ├─ utils.py
│  └─ session_data.npz
├─ tests/             # unit & integration tests
├─ requirements.txt
└─ LICENSE
```

---

## 🛠️ Extending & Customizing

1. **Add New Metrics**

   * Define a function in `metrics.py` → integrate into `compute_all_metrics()`.
2. **Swap Models**

   * YOLO: pass a different `--weights` file (e.g., custom fine-tuned).
   * Mediapipe: adjust `MODEL_COMPLEXITY` (0/1/2).
3. **Batch Mode**

   * Provide a folder to `--source` → auto-iterates over all videos.

---

## 🤝 Contributing

1. Fork the repo & create a feature branch
2. Write tests & ensure `pytest` passes
3. Submit a PR with clear description & linked issue

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](../LICENSE) for details.

```
```
