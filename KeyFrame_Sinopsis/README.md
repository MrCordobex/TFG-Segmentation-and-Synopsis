# Project: Clustering and Visual Analysis of Video

This repository contains a set of tools to analyze a video using *deep learning*, *dimensionality reduction*, and *clustering* techniques, with the goal of generating a representative synopsis from the most relevant frames.

## General Structure

The pipeline includes:

1. **Frame extraction** from a video.
2. **Feature embedding extraction** using a pretrained neural network (ResNet101).
3. **Similarity matrix computation** between embeddings (cosine or RBF).
4. **Dimensionality reduction** (MDS or FastMDS).
5. **Clustering** of projected embeddings (KMeans or HDBSCAN).
6. **Object detection** on clustered frames using YOLOv8.
7. **Temporal analysis of clusters** using time series and peak detection.
8. **Selection and visualization** of representative frames.
9. **Generation of a summary video**.

---

## Step-by-Step Usage

### 1. Extract frames from a video

```python
video_path = "/path/to/video.mp4"
output_folder = "/path/to/frames"
video_to_frames(video_path, output_folder, prefix='frame', skip=1)
```

### 2. Group visually similar frames

```python
folder = "/path/to/frames"
similarity = 'rbf'           # or 'cosine'
reducction = 'FastMDS'       # or 'MDS'
clustering = 'HDBSCAN'       # or 'KMEANS'

df = grouping(folder, similarity, reducction, clustering,
              n_clusters=2,
              plot_similitud=False, plot_reduction=False,
              plot_clustering=True, plot_images=True)
```

### 3. Remove noise (unlabeled or spurious frames)

```python
df = clean_noise(df)
```

### 4. Apply YOLO detection model

```python
model_path = "/path/to/model.pt"
input = "/path/to/frames"
output = "/path/to/yolo_frames"
filter = [1,3,4,8]  # Classes to detect
threshold = 0.2     # Confidence threshold

df_inferenced = yolo_prediction(df, model_path, input, output, filter, conf_threshold=threshold)
```

### 5. Generate visual synopsis (representative frames per cluster)

```python
yolo_folder = output
output_folder = "/path/to/synopsis"
frames = sinopsis(df_inferenced, yolo_folder, output_folder, plot_series=True, plot_frames=True)
```

### 6. Convert selected frames into summary video

```python
frames_folder = output_folder
output_video = "/path/to/output.mp4"
fps = 30
frames_to_video(frames_folder, output_video, fps=fps, prefix='frame')
```

---

## Main Dependencies

* `torch`, `torchvision`, `PIL`, `scikit-learn`, `hdbscan`
* `ultralytics` (YOLOv8)
* `matplotlib`, `numpy`, `opencv-python`, `pandas`, `tqdm`, `scipy`

---

## Code Organization

* `utils/clustering.py` → KMeans and HDBSCAN functions.
* `utils/cosenoRBF.py` → embeddings, similarity matrix, normalization.
* `utils/dimesionalreduction.py` → MDS and FastMDS.
* `utils/functions.py` → noise cleaning.
* `utils/grouping.py` → main grouping function.
* `utils/inference.py` → YOLO inference and `frame_score` calculation.
* `utils/sinopsis.py` → time series, peak selection, and summary generation.
* `utils/video.py` → video-to-frame and frame-to-video conversion.

---

## Author

Developed by **Pedro Martínez-Huertas** · pedromarhuer03@gmail.com
Authorized by the company **EONSEA** as part of an internal computer vision project.

---
## 🤝 Contributing

Feel free to open issues or pull requests to improve the notebooks, add new features, or fix bugs.

---

## License

Distributed under the terms of the MIT license. See the `LICENSE` file for more details.
