# Hyperparameters

## 1. Basic Task/Runtime Parameters

* **task**: `segment`
  Indicates the task: `segment` (segmentation), `detect` (object detection), `classify` (classification), etc.
* **mode**: `train`
  Execution mode: `train` for training, `val` for validation, `predict` for inference, etc.
* **model**: `yolo11n-seg.pt`
  Base model to use. Can be a pre-trained checkpoint or specific architecture (e.g., `yolov8n-seg.pt`, `yolo11n-seg.pt`, etc.).
* **data**: `dataset_procesado/data.yaml`
  Data configuration file defining train/val/test paths and class names.
* **epochs**: `40`
  Number of epochs (full dataset iterations) for training.
* **time**: `None`
  Optional time limit or tracking. `None` means no explicit time limit.
* **patience**: `100`
  Early-stop patience: epochs without validation improvement before stopping.
* **batch**: `16`
  Batch size per training iteration.
* **imgsz**: `640`
  Input image size (e.g., 640×640).
* **save**: `True`
  Whether to save outputs (weights, logs, plots).
* **save\_period**: `-1`
  Epoch interval for saving intermediate weights (`-1` = only at end).
* **cache**: `False`
  Cache images/features in RAM for faster training (higher memory usage).
* **device**: `None`
  Compute device (CPU/GPU). `None` lets YOLO auto-select (typically GPU if available).
* **workers**: `8`
  Number of DataLoader workers for parallel data loading.
* **project**: `None`
  Base folder to store results (default `runs/segment` if `None`).
* **name**: `train2`
  Experiment name; results go to `runs/segment/train2` (or `project/name`).
* **exist\_ok**: `False`
  Allow overwriting existing results folder if `True`, else error.
* **pretrained**: `True`
  Load pretrained weights if `True`, else random init.
* **optimizer**: `auto`
  Optimizer choice (e.g., `auto`, `sgd`, `adam`, `AdamW`).
* **verbose**: `True`
  Show detailed training logs.
* **seed**: `0`
  Random seed for reproducibility.
* **deterministic**: `True`
  Enforce deterministic operations where possible.
* **single\_cls**: `False`
  Treat all classes as one (useful for single-object datasets).
* **rect**: `False`
  Rectangular batching (maintain original aspect ratios).
* **cos\_lr**: `False`
  Use cosine learning rate scheduler if `True` (default is linear).
* **close\_mosaic**: `10`
  Disable mosaic augmentation this many epochs before the end.
* **resume**: `False`
  Resume training from last checkpoint if `True`.
* **amp**: `True`
  Automatic Mixed Precision (float16/float32) for faster training.
* **fraction**: `1.0`
  Fraction of dataset to use (1.0 = full dataset).
* **profile**: `False`
  Collect performance metrics if `True`.
* **freeze**: `None`
  Layers to freeze (e.g., `10` to freeze first 10 layers).
* **multi\_scale**: `False`
  Randomly vary input size per batch (±50%).
* **overlap\_mask**: `True`
  Allow overlapping masks in segmentation output.
* **mask\_ratio**: `4`
  Internal downsampling factor for segmentation masks.
* **dropout**: `0.0`
  Dropout rate in the model.
* **val**: `True`
  Run validation every epoch if `True`.
* **split**: `val`
  Which split to use for validation (`val`, `test`, etc.).
* **save\_json**: `False`
  Save validation results in COCO JSON format.
* **save\_hybrid**: `False`
  Save hybrid annotations (boxes + masks).
* **conf**: `None`
  Confidence threshold for inference/validation.
* **iou**: `0.7`
  IoU threshold for NMS.
* **max\_det**: `300`
  Maximum detections per image.
* **half**: `False`
  Use half precision (FP16) during inference.
* **dnn**: `False`
  Use OpenCV DNN for inference if `True`.
* **plots**: `True`
  Generate and save training plots.
* **source**: `None`
  Input source for `predict` mode (images/videos).
* **vid\_stride**: `1`
  Frame stride for video inference.
* **stream\_buffer**: `False`
  Buffering for live video streams.
* **visualize**: `False`
  Internal feature visualization.
* **augment**: `False`
  Extra augmentation during inference/validation.
* **agnostic\_nms**: `False`
  Class-agnostic NMS if `True`.
* **classes**: `None`
  Filter by class indices.
* **retina\_masks**: `False`
  Generate full-resolution segmentation masks.
* **embed**: `None`
  Embed parameter for visualizers.
* **show**: `False`
  Display predictions in a window.
* **save\_frames**: `False`
  Save inference frames.
* **save\_txt**: `False`
  Save YOLO TXT results.
* **save\_conf**: `False`
  Save confidences in TXT results.
* **save\_crop**: `False`
  Save cropped detected objects.
* **show\_labels**: `True`
  Show class labels.
* **show\_conf**: `True`
  Show confidence values.
* **show\_boxes**: `True`
  Show bounding boxes.
* **line\_width**: `None`
  Line width for bounding boxes.
* **format**: `torchscript`
  Export format (e.g., `torchscript`, `onnx`).
* **keras**, **optimize**, **int8**, **dynamic**, **simplify**, **opset**, **workspace**, **nms**: `False`
  Advanced export/optimization options.

## 2. Optimization Hyperparameters

* **lr0**: `0.01`
  Initial learning rate.
* **lrf**: `0.01`
  Final LR factor: LR decays to `lr0 * lrf`.
* **momentum**: `0.937`
  Momentum for SGD/Adam.
* **weight\_decay**: `0.0005`
  L2 weight decay.
* **warmup\_epochs**: `3.0`
  Warm-up period for LR ramp-up.
* **warmup\_momentum**: `0.8`
  Initial momentum during warm-up.
* **warmup\_bias\_lr**: `0.1`
  Initial bias LR during warm-up.
* **box**: `7.5`, **cls**: `0.5`, **dfl**: `1.5`, **pose**: `12.0`, **kobj**: `1.0`
  Loss balance factors for box, cls, DFL, pose, and key-object.
* **nbs**: `64`
  Nominal batch size reference for LR scaling.

## 3. Data Augmentation Hyperparameters

* **hsv\_h**: `0.015`, **hsv\_s**: `0.7`, **hsv\_v**: `0.4`
  HSV augmentation ranges for hue, saturation, value.
* **degrees**: `0.0`
  Random rotation range in degrees.
* **translate**: `0.1`
  Random translation fraction (±10%).
* **scale**: `0.5`
  Random scale factor (±50%).
* **shear**: `0.0`
  Random shear angle.
* **perspective**: `0.0`
  Perspective distortion.
* **flipud**: `0.0`
  Vertical flip probability.
* **fliplr**: `0.5`
  Horizontal flip probability.
* **bgr**: `0.0`
  BGR-to-RGB channel swap probability.
* **mosaic**: `1.0`
  Mosaic augmentation probability.
* **mixup**: `0.0`
  MixUp augmentation probability.
* **copy\_paste**: `0.0`
  Copy-paste augmentation probability.
* **copy\_paste\_mode**: `flip`
  Copy-paste mode.
* **auto\_augment**: `randaugment`
  AutoAugment strategy.
* **erasing**: `0.4`
  Random erasing probability.
* **crop\_fraction**: `1.0`
  Random crop fraction.

## 4. Other Parameters

* **cfg**: `None`
  Extra network config file (often unused).
* **tracker**: `botsort.yaml`
  Tracker config for video tracking.
* **save\_dir**: `runs/segment/train2`
  Final results directory (`project + name`).
