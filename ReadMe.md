# Vehicle Re-Identification Pipeline

This project implements a full pipeline for **vehicle re-identification** in real time using computer vision. The pipeline project simulates running the pipeline on any two "live" video streams, where vehicles from the first video are re-identified to the second. It includes:
- Object detection (YOLO)
- Tracking (ByteTrack)
- Region-based vehicle crop filtering (Strategy to take snapshots of the vehicles in specific areas and specific amount of times)
- Feature extraction (custom and pretrained ReID model with ResNet-50 backbone)
- Vector database with LanceDB
- Re-identification matching
- Visual analytics with frame annotation

![](gif/reid_demo.gif)


---

## 🚀 Setup Instructions

1. **Install dependencies** (recommend using a virtual environment):

```bash
pip install -r requirements.txt

```

Requirements include:

    opencv-python

    ultralytics (for YOLOv8)

    torch, torchvision

    lancedb

    numpy


2. **Run the Demo**
   
```sh

   python main.py

```

   Or add custom arguments. the default arguments are as follows:

```sh

   python main.py \
    --video_path1 videos/vdo4.avi \
    --video_path2 videos/vdo1.avi \
    --roi_path1 videos/vdo4_roi.png \
    --roi_path2 videos/vdo1_roi.png \
    --detection_model yolov8n.pt \
    --device cuda \
    --scnd_video_offset 0 \
    --reID_features_size 256 \
    --reID_features_expire 100 \
    --crop_zone_rows_vid1 7 \
    --crop_zone_cols_vid1 6 \
    --crop_zone_area_bottom_left_vid1 0 1000 \
    --crop_zone_area_top_right_vid1 1750 320 \
    --crop_zone_rows_vid2 7 \
    --crop_zone_cols_vid2 6 \
    --crop_zone_area_bottom_left_vid2 200 900 \
    --crop_zone_area_top_right_vid2 1750 320
```

## Command-line Arguments

| Argument                 | Type   | Default               | Description                                                                                               |
| ------------------------ | ------ | --------------------- | --------------------------------------------------------------------------------------------------------- |
| `--video_path1`          | `str`  | `videos/vdo4.avi`     | Path to the **first video** (vehicles to re-identify **FROM**).                                           |
| `--video_path2`          | `str`  | `videos/vdo1.avi`     | Path to the **second video** (vehicles to re-identify **TO**).                                            |
| `--roi_path1`            | `str`  | `videos/vdo4_roi.png` | Path to the ROI mask for video 1. If not specified, auto-detected based on video name.                    |
| `--roi_path2`            | `str`  | `videos/vdo1_roi.png` | Path to the ROI mask for video 2. If not specified, auto-detected based on video name.                    |
| `--detection_model_path` | `str`  | `'yolov8x.pt'`        | YOLO model file to use for detection. One of: `'yolov8x.pt'`, `'yolov8l.pt'`, `'yolov5su.pt'`.            |
| `--device`               | `str`  | `'cuda'`              | Computation device to use: `'cuda'` or `'cpu'`.                                                           |
| `--scnd_video_offset`    | `int`  | `0`                   | Number of **frames to delay** processing the second video. Useful for syncing when vehicles appear later. |
| `--reID_features_size`   | `int`  | `256`                 | Dimensionality of the feature vectors used for Re-ID.                                                     |
| `--debug`                | `bool` | `True`                | Whether to show debugging info (e.g. crop zones).                                                         |
| `--reID_features_expire` | `int`  | `100`                 | How many frames a feature stays in memory before being removed from the database.                         |

🟩 Crop Zone Settings – Video 1

| Argument                            | Type    | Default       | Description                                            |
| ----------------------------------- | ------- | ------------- | ------------------------------------------------------ |
| `--crop_zone_rows_vid1`             | `int`   | `7`           | Number of rows in crop zone grid.                      |
| `--crop_zone_cols_vid1`             | `int`   | `6`           | Number of columns in crop zone grid.                   |
| `--crop_zone_area_bottom_left_vid1` | `tuple` | `(0, 1000)`   | Bottom-left `(x, y)` coordinate of the crop zone area. |
| `--crop_zone_area_top_right_vid1`   | `tuple` | `(1750, 320)` | Top-right `(x, y)` coordinate of the crop zone area.   |

🟦 Crop Zone Settings – Video 2

| Argument                            | Type    | Default       | Description                                            |
| ----------------------------------- | ------- | ------------- | ------------------------------------------------------ |
| `--crop_zone_rows_vid2`             | `int`   | `7`           | Number of rows in crop zone grid.                      |
| `--crop_zone_cols_vid2`             | `int`   | `6`           | Number of columns in crop zone grid.                   |
| `--crop_zone_area_bottom_left_vid2` | `tuple` | `(200, 900)`  | Bottom-left `(x, y)` coordinate of the crop zone area. |
| `--crop_zone_area_top_right_vid2`   | `tuple` | `(1750, 320)` | Top-right `(x, y)` coordinate of the crop zone area.   |

## Using Your Own Videos

To use your own surveillance videos for vehicle re-identification, follow these steps:

1. Replace the video files
Place your video files in the `videos/` directory or provide custom paths using:

```sh
--video_path1 path/to/your_first_video.avi
--video_path2 path/to/your_second_video.avi
```
2. Provide corresponding ROI masks
   
Each video should have a corresponding ROI image (e.g., `video1_roi.png`) highlighting the region where vehicles appear. If omitted, the script will attempt to locate one automatically based on the video name.

3. Adjust crop zones

Modify the crop zone grid and bounding box reidentification area to match your camera view using:

```sh
--crop_zone_rows_vid1 7
--crop_zone_cols_vid1 6
--crop_zone_area_bottom_left_vid1 (x, y)
--crop_zone_area_top_right_vid1 (x, y)

```
Do the same for `vid2`.

4. Set detection offset (optional)

If vehicles in the second video appear after a delay, use `--scnd_video_offset` to delay processing the second stream by a number of frames.

## 🔧 Project Structure

├── main.py # Main demo script
├── detector.py # VehicleDetector class (YOLO + ByteTrack)
├── cropZoneFilter.py # CropZoneFilter class for area-specific crops
├── featureExtractor.py # ExtractingFeatures class (ReID feature extraction)
├── lanceDBOperator.py # Feature vector storage and search using LanceDB
├── trackerStateHelper.py # ReIDController: matching and ID assignment
├── visualizer.py # Annotates frames with detection + tracking + ReID
├── videos/ # Input videos and ROI masks
└── requirements.txt # Python dependencies

## 🧠 Core Components

`🔍 VehicleDetector`

- Wraps OpenCV video reader
- Loads YOLOv8 model
- Uses ByteTrack for ID-based multi-object tracking 
- Optionally masks background using ROI images

`✂️ CropZoneFilter`

- Divides a specified region into a grid of zones
- Filters and crops detections within zones
- Provides the logic to have vehicles be saved around 4-8 times in different positions to improve ReID accuracy, but not have to save in every frame.

`🧬 ExtractingFeatures`

- Extracts ReID feature vectors from image crops
- Uses a pretrained CNN (e.g., ResNet50) with custom embedding size (default: 256)

`💽 LanceDBOperator`

- Stores features in a local LanceDB table
- Supports vector search (top-1 matching)
- Features auto-expire after configurable max_age

`🔁 ReIDController`

- Matches vehicles from video 2 against database of vehicles from video 1
- Assigns matched IDs to tracked objects

`🎨 Visualizer`

Annotates frame with:

- Tracker ID
- Class name
- Detection confidence (vid1) and similarity score (vid2)


## Acknowledgements

This work was supported by Chips Joint Undertaking (Chips JU) 
in **EdgeAI** “Edge AI Technologies for Optimised Performance 
Embedded Processing” project, grant agreement No 101097300. at **EDI** (Institute of Electronics and Computer Science), Latvia.

Some parts of the codebase, particularly for the ReID model training, were adapted from [regob/vehicle_reid ](https://github.com/regob/vehicle_reid) – an open-source vehicle re-identification repository. Their work served as a valuable foundation for the model development in this project.

## Papers

The work is described in detail in the *Multi-Step Object Re-Identification on Edge Devices: A Pipeline for Vehicle Re-Identification* paper presented at the 2024 EEAI EdgeAI conference.

    @inproceedings{17880_2025,
    author = {Tomass Zutis and Peteris Racinskis and Anzelika Bureka and Janis Judvaitis and Janis Arents and and Modris Greitans},
    title = {Multi-Step Object Re-Identification on Edge Devices: A Pipeline for Vehicle Re-Identification},
    journal = {TBA},
    year = {2025}
    }

https://www.edi.lv/en/publications/multi-step-object-re-identification-on-edge-devices-a-pipeline-for-vehicle-re-identification-2/