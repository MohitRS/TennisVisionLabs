# TennisVisionLabs

**AI-Powered Tennis Video Analytics Prototype**

TennisVisionLabs is a proof-of-concept framework demonstrating core AI features for tennis video analysis. Through a series of development steps, we’ll build independent modules that can later be integrated into a full-stack product rivaling solutions like SwingVision.

---

## 📦 Directory Structure

```plaintext
TennisVisionLabs/
├── .gitignore
├── README.md
├── requirements.txt
├── setup.cfg
│
├── data/
│   ├── raw/          # Unedited tennis clips
│   └── processed/    # Resized / annotated samples
│
├── notebooks/
│   └── step1_setup.ipynb   # Environment check & sample video load
│
├── src/
│   ├── player_detection/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── utils.py
│   ├── ball_tracking/
│   ├── pose_estimation/
│   ├── stroke_classification/
│   ├── metrics/
│   └── analytics/
│
├── tests/             # pytest suites for each module
└── .vscode/           # VS Code launch/settings/tasks
```

---

## 🔧 Prerequisites

* **Python 3.10** (MediaPipe supports 3.7–3.10)
* **Git**
* **Visual Studio C++ Build Tools** (for native extensions)

---

## ⚙️ Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/MohitRS/TennisVisionLabs.git
   cd TennisVisionLabs
   ```

2. **Create & activate a virtual environment**

   ```bash
   py -3.10 -m venv .venv
   # Windows (cmd.exe)
   .venv\Scripts\activate.bat

   # or PowerShell
   .\.venv\Scripts\Activate.ps1
   ```

3. **Upgrade packaging tools & install dependencies**

   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Verify environment**

   ```bash
   pip install notebook
   jupyter notebook notebooks/step1_setup.ipynb
   ```

---

## 🚀 Development Steps

We’ll deliver seven standalone modules—each in its own folder under `src/`—to prove core capabilities:

1. **Player & Ball Detection**
   – Fine-tune YOLOv8 to localize players and tennis balls.
2. **Ball Tracking**
   – Implement a Kalman-filter tracker to maintain ball identity across frames.
3. **Pose Estimation**
   – Extract 2D keypoints (MediaPipe or alternative) and overlay skeletons.
4. **Stroke Classification**
   – Train a sequence model (e.g. LSTM/SVM) to label forehand, backhand, serve, etc.
5. **Speed & Spin Metrics**
   – Compute player/ball speeds and approximate spin via trajectory analysis.
6. **Session Analytics Report**
   – Auto-generate PDF/HTML summaries: stroke counts, speed distributions, rally lengths.
7. **Advanced Visualizations**
   – Create shot-placement heatmaps, footwork distance plots, and other insights.

Each step lives in its own subdirectory in `src/` with `train.py`, `inference.py` (or equivalent), and `utils.py`. Tests go under `tests/`.

---

## 💡 Usage

* **Notebooks**

  * `notebooks/step1_setup.ipynb` verifies imports and sample video loading.
  * Subsequent notebooks will demo each module independently.

* **Scripts**

  ```bash
  # Example: train player detector
  python src/player_detection/train.py --help

  # Example: run inference
  python src/player_detection/inference.py data/raw/rally1.mp4
  ```

---

## 🤝 Contributing

1. **Branching**

   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Coding & Testing**

   * Follow PEP 8
   * Write tests in `tests/`
3. **Pull Requests**

   * Push your branch
   * Open a PR against `main`
   * Request a review

---

## 📜 License

Released under the [MIT License](LICENSE).

---

Let’s get started and build a world-class tennis analytics prototype!
