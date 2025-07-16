# 🎾 TennisVisionLabs

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE) [![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

**AI-Powered Tennis Video Analytics Prototype**

TennisVisionLabs is a modular proof-of-concept framework demonstrating core AI features for tennis video analysis. Each component—from player & ball detection to pose-estimation and session analytics—lives in its own folder and can be developed, tested and integrated independently.

---

## 🚀 Table of Contents

1. [Status & Roadmap](#-status--roadmap)  
2. [Directory Structure](#-directory-structure)  
3. [Features & Methodologies](#-features--methodologies)  
4. [Installation](#-installation)  
5. [Usage](#-usage)  
6. [Project Structure](#-project-structure)  
7. [Contributing](#-contributing)  
8. [License](#-license)  

---

## ✨ Status & Roadmap

| Module                         | Status      | Key Scripts                                |
|--------------------------------|-------------|--------------------------------------------|
| ✅ Player & Ball Detection       | Completed   | `src/player_detection/train.py`  `inference.py` |
| ✅ Ball Tracking                  | Completed   | `src/ball_tracking/track.py`               |
| ✅ Pose Estimation                | Completed   | `src/pose_estimation/media_pipe_pose.py`  `yolo8_pipeline.py` |
| ⬜ Stroke Classification          | Planned     | `src/stroke_classification/…`              |
| ⬜ Speed & Spin Metrics           | Planned     | `src/metrics/…`                            |
| ⬜ Session Analytics Report       | Planned     | `src/analytics/report.py`                  |
| ⬜ Advanced Visualizations        | Planned     | `src/analytics/visuals.py`                 |

---

## 🏗 Features & Methodologies

### 1. Player & Ball Detection  
- **Approach**: Fine-tuned YOLOv8  
- **Output**: Bounding boxes for players & balls  

### 2. Ball Tracking  
- **Approach**: Kalman filter + IoU data association  
- **Output**: Trajectories with consistent IDs  

### 3. Pose Estimation  
Two interchangeable pipelines:

| Pipeline            | Library      | Device       | Script                         |
|---------------------|--------------|--------------|--------------------------------|
| **MediaPipe**       | `mediapipe`  | CPU-friendly | `media_pipe_pose.py`           |
| **YOLOv8+Pose**     | `ultralytics`| GPU/CPU      | `yolo8_pipeline.py`            |

**Key steps**:
1. **Frame Extraction** (`--stride`)  
2. **Keypoint Detection** (BlazePose or YOLOv8-pose)  
3. **Rally & Stroke Segmentation** via velocity peaks  
4. **Metric Computation** (`metrics.py`):  
   - Stability & balance (%)  
   - Elbow ROM (min/median/max)  
   - Symmetry index (%)  
   - Stance width & torso twist velocity  
5. **Reporting**:  
   - `report.txt` summary  
   - Optional `visualized.mp4` overlay  
   - `session_data.npz` for downstream analysis  

---

## ⚙️ Installation

```bash
# Clone  
git clone https://github.com/MohitRS/TennisVisionLabs.git
cd TennisVisionLabs

# Virtual environment  
python3.10 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows

# Dependencies  
pip install --upgrade pip
pip install -r requirements.txt
````

---

## 🏃 Usage

### Player & Ball Detection

```bash
python src/player_detection/train.py --data data/player_detection.yaml
python src/player_detection/inference.py data/raw/rally1.mp4 --weights yolov8n.pt
```

### Ball Tracking

```bash
python src/ball_tracking/track.py \
  --detections runs/player_detection/…/pred.csv \
  --output runs/ball_tracking/trajectory.mp4
```

### Pose Estimation

#### MediaPipe (CPU)

```bash
python src/pose_estimation/media_pipe_pose.py \
  --source data/raw/rally1_junaidraw.mp4 \
  --stride 1 \
  --visualize
```

#### YOLOv8+Pose (GPU/CPU)

```bash
python src/pose_estimation/yolo8_pipeline.py \
  --source data/raw/rally1_junaidraw.mp4 \
  --weights yolov8n-pose.pt \
  --conf 0.4 \
  --visualize \
  --save-csv
```

---

## 📁 Project Structure

```plaintext
TennisVisionLabs/
├── data/
│   ├── raw/              # raw tennis videos (.mp4, .avi)
│   └── processed/        # extracted frames, artifacts
│
├── notebooks/            # demo & exploratory notebooks
│   └── step1_setup.ipynb
│
├── src/                  # core modules
│   ├── player_detection/
│   ├── ball_tracking/
│   ├── pose_estimation/
│   ├── stroke_classification/
│   ├── metrics/
│   └── analytics/
│
├── tests/                # pytest suites
├── requirements.txt
├── setup.cfg
└── README.md             # <— you are here
```

---

## 🤝 Contributing

1. Fork → feature branch
2. Code & test (`pytest`)
3. PR review & merge

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

