# YOLO Project

This repository contains several notebooks for training, evaluating, comparing, and running YOLOv8 models in real time. Below is an overview of each notebook and the requirements needed.

---

## 📋 Prerequisites

1. **Python**: version 3.8 or higher.
2. **Pip**: Python package manager.

### Libraries and Dependencies

Install the required libraries with:

```bash
pip install ultralytics opencv-python-headless numpy matplotlib pyyaml
```

> **Note**: If you need OpenCV GUI support, install `opencv-python` instead of `opencv-python-headless`.

---

## 📂 Repository Structure

```text
├── Training.ipynb       # Notebook for training the model
├── Evaluation.ipynb     # Notebook for validating the model on the test set
├── Comparison.ipynb     # Notebook for comparing outputs of two models
├── Real_time.ipynb      # Notebook for real-time inference (webcam)
└── dataset/             # Folder containing images and labels
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

---

## 🏋️‍♂️ Training.ipynb

1. **YAML Generation**:

   * Defines the dataset root (`dataset/`) and subfolders (`images/train`, `images/val`, `images/test`).
   * Maps class indices to human-readable names.
   * Saves the configuration to `dataset/data.yaml`.

2. **Training**:

   * Loads a pre-trained model (`yolo11n-seg.pt`).
   * Runs `model.train()` with parameters:

     * `data`: path to the YAML file.
     * `epochs`: number of epochs (default 40).
     * `imgsz`: image size (default 640).
     * `plots`: whether to generate training plots.

> **Output**: The `runs/train/` directory with weights, logs, and training curves.

3. **Hyperparameters**:

   * YOLO supports numerous training hyperparameters (e.g., learning rate, batch size, optimizer settings, augmentation parameters).
   * For a detailed description of all available hyperparameters and recommended values, see `YOLO_hyperparameters.md`.

---

## ✅ Evaluation.ipynb

1. **`evaluate_model` Function**:

   * Checks for existence of image and label directories.
   * Constructs a temporary YAML pointing to `images/test` for validation.
   * Loads the model from `weights_path`.
   * Runs `model.val(data=tmp_yaml, iou=iou_threshold)`.
   * Deletes the temporary YAML file after validation.

2. **Usage**:

   ```python
   evaluate_model(
       WEIGHTS="model.pt",
       IMAGES_TEST="dataset/images/test",
       LABELS_TEST="dataset/labels/test",
       CLASS_NAMES={...},
       iou_threshold=0.6
   )
   ```

> **Output**: The `runs/val/` directory with metrics (mAP, precision, recall) and visualizations.

---

## 📊 Comparison.ipynb

1. **Objective**: Compare predictions from two models on the same images.

2. **Key Functions**:

   * `visualize_predictions`: saves and reloads image with detections, returns the image and a summary of class counts.
   * `process_images_two_models`: iterates over images in two folders, runs inference with both models, and displays side-by-side results.

3. **Usage**:

   ```python
   model1 = YOLO("model_1.pt")
   model2 = YOLO("model_2.pt")
   process_images_two_models(
       "folder_1_path",  # original images
       "folder_2_path",  # preprocessed images
       model1,
       model2
   )
   ```

> **Output**: A Matplotlib figure with one row per image and two columns (Model 1 vs. Model 2).

---

## 🎥 Real\_time.ipynb

1. **`run_yolo` Function**:

   * Arguments:

     * `model_path`: path to the `.pt` file (detection or segmentation).
     * `mode`: "detect" or "segment".
     * `confidence_threshold`: minimum confidence threshold.
     * `device`: camera index (default 0).

2. **Workflow**:

   * Loads the specified model.
   * Opens the webcam using OpenCV.
   * Captures frames in a loop:

     * Runs inference on each frame.
     * Draws bounding boxes (`detect`) or masks and boxes (`segment`).
     * Displays the video in a window.
     * Press `Q` to exit.

3. **Example**:

   ```python
   run_yolo(
       model_path="yolov8n-seg.pt",
       mode="segment",
       confidence_threshold=0.5,
       device=0
   )
   ```

---

## 🚀 Getting Started

1. Clone the repository.
2. Install dependencies.
3. Prepare your dataset following the structure above.
4. Run the notebooks in the following order:

   1. `Training.ipynb`
   2. `Evaluation.ipynb`
   3. `Comparison.ipynb` (optional)
   4. `Real_time.ipynb` (optional)

---
## Author

Developed by **Pedro Martínez-Huertas** · pedromarhuer03@gmail.com
Authorized by the company **EONSEA** as part of an internal computer vision project.

---

## 🤝 Contributing

Feel free to open issues or pull requests to improve the notebooks, add new features, or fix bugs.

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
