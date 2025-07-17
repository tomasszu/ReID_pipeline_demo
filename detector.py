import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import os

class VehicleDetector:
    def __init__(self, video_path: str, class_ids=None, model_path="yolov8x.pt", device="cuda", roi_path=None):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path).to(device)
        self.tracker = sv.ByteTrack()
        self.device = device

        self.class_ids = class_ids if class_ids else [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = self.model.model.names
        self.conf_threshold = 0.6

        self.roi_mask = self._load_roi(roi_path, video_path)

    def _load_roi(self, roi_path, video_path):
        # If explicitly given, try to load
        if roi_path and os.path.exists(roi_path):
            print(f"Using provided ROI from: {roi_path}")
            return cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        # Otherwise try to derive from video path
        base_path, _ = os.path.splitext(video_path)
        auto_roi_path = base_path + "_roi.png"

        if os.path.exists(auto_roi_path):
            print(f"Using auto-detected ROI from: {auto_roi_path}")
            # Ensure the ROI is grayscale
            return cv2.imread(auto_roi_path, cv2.IMREAD_GRAYSCALE)

        print(f" No auto found ROI with filename {auto_roi_path}. Using full frame.")
        return None


    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def process_frame(self, frame):

        if self.roi_mask is not None:
            roi_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        else:
            roi_frame = frame.copy()

        results = self.model(roi_frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Filter classes & confidence
        detections = detections[np.isin(detections.class_id, self.class_ids)]
        detections = detections[np.greater(detections.confidence, self.conf_threshold)]



        # Update with tracker
        detections = self.tracker.update_with_detections(detections)

        return detections, frame

    def release(self):
        self.cap.release()
