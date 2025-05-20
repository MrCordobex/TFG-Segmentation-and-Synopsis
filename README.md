# Multi-Project Repository: Video Analysis and YOLO

This monorepo contains two main subprojects focused on video analysis:

1. **KeyFrame\_Synopsis**: Clustering and visual synopsis generation from video frames using deep learning, dimensionality reduction, and clustering.
2. **YOLO**: Training, evaluation, comparison, and real-time inference with YOLOv8 models for object detection and segmentation.

---

## 📋 Prerequisites

* **Python 3.8+**
* **Pip** (Python package manager)

Install core dependencies used across both projects:

```bash
pip install torch torchvision ultralytics opencv-python-headless numpy matplotlib pyyaml scikit-learn hdbscan pandas tqdm scipy
```

> If you need GUI support for OpenCV, replace `opencv-python-headless` with `opencv-python`.

---

## 📂 Repository Structure

```text
├── KeyFrame_Synopsis/      # Video clustering and synopsis project
│   ├── main.ipynb          # Main notebook for the full pipeline
│   ├── README.md           # Project-specific README
│   └── utils/              # Utility modules
│       ├── clustering.py
│       ├── cosenoRBF.py
│       ├── dimensionalreduction.py
│       ├── functions.py
│       ├── grouping.py
│       ├── inference.py
│       ├── sinopsis.py
│       └── video.py
├── YOLO/                   # YOLOv8 training and inference project
│   ├── Training.ipynb      # Notebook for model training
│   ├── Evaluation.ipynb    # Notebook for model validation
│   ├── Comparison.ipynb    # Notebook to compare two models' outputs
│   ├── Real_time.ipynb     # Notebook for webcam inference
│   ├── README.md           # Project-specific README
│   └── YOLO_hyperparameters.md # Detailed hyperparameters reference
└── README.md               # This combined README
```

---

## 🔧 KeyFrame\_Synopsis

A pipeline to generate a representative video synopsis:

1. **Frame Extraction**

   ```python
   video_to_frames(video_path, output_folder, prefix='frame', skip=1)
   ```
2. **Embedding Extraction & Similarity**

   ```python
   df = grouping(
       frames_folder,
       similarity='rbf',      # or 'cosine'
       reduction='FastMDS',    # or 'MDS'
       clustering='HDBSCAN',   # or 'KMEANS'
       ...
   )
   ```
3. **Noise Cleaning**

   ```python
   df = clean_noise(df)
   ```
4. **YOLO Inference on Clusters**

   ```python
   df_inf = yolo_prediction(
       df, model_path, input_folder, output_folder,
       filter=[1,3,4], conf_threshold=0.2
   )
   ```
5. **Synopsis Generation**

   ```python
   frames = sinopsis(df_inf, yolo_folder, synopsis_folder, plot_series=True)
   ```
6. **Video Synthesis**

   ```python
   frames_to_video(synopsis_folder, output_video, fps=30)
   ```

**Dependencies**: `torch`, `torchvision`, `scikit-learn`, `hdbscan`, `matplotlib`, `numpy`, `opencv-python`, `pandas`, `scipy`

---

## 🏋️‍♂️ YOLO Project

Workflows for YOLOv8:

### 1. Training (Training.ipynb)

* Generates `dataset/data.yaml` for your `dataset/images` and `dataset/labels`.
* Runs:

  ```python
  model.train(
    data='dataset/data.yaml',
    epochs=40, imgsz=640, plots=True
  )
  ```
* **Hyperparameters**: See `YOLO_hyperparameters.md` for full list.

> **Output**: `runs/train/` with weights, logs, and training curves.

### 2. Evaluation (Evaluation.ipynb)

* Validates on `dataset/images/test` & `dataset/labels/test` via temporary YAML.
* Usage:

  ```python
  evaluate_model(
    WEIGHTS='model.pt',
    IMAGES_TEST='dataset/images/test',
    LABELS_TEST='dataset/labels/test',
    CLASS_NAMES={...}
  )
  ```

> **Output**: `runs/val/` with mAP, precision, recall, and plots.

### 3. Model Comparison (Comparison.ipynb)

* Side-by-side inference of two models on identical images.
* Usage:

  ```python
  process_images_two_models(
    'folder1', 'folder2', model1, model2
  )
  ```

### 4. Real-Time Inference (Real\_time.ipynb)

* Webcam loop for `detect` or `segment` mode.
* Usage:

  ```python
  run_yolo(
    model_path='yolov8n-seg.pt', mode='segment',
    confidence_threshold=0.5, device=0
  )
  ```

**Dependencies**: `ultralytics`, `opencv-python-headless`, `numpy`, `matplotlib`, `pyyaml`
---

## Author

Developed by **Pedro Martínez-Huertas** · pedromarhuer03@gmail.com
Authorized by the company **EONSEA** as part of an internal computer vision project.
---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or pull request in the relevant subfolder.

---

## 📄 License

Distributed under the MIT License. See the `LICENSE` file for details.
