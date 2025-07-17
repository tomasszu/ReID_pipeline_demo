import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np

class VehicleDetector:
    def __init__(self, video_path: str, class_ids=None, model_path="yolov8x.pt", device="cuda", roi_path=None):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path).to(device)
        self.tracker = sv.ByteTrack()
        self.device = device
        self.roi_mask = None

        self.class_ids = class_ids if class_ids else [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = self.model.model.names
        self.conf_threshold = 0.6

        if roi_path:
            self.roi_mask = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
            if self.roi_mask is None:
                raise ValueError(f"ROI mask not found at {roi_path}")
            self.roi_mask = (self.roi_mask > 0).astype(np.uint8)  # binary mask



    def read_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def process_frame(self, frame):

        if self.roi_mask is not None:
            roi_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)

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
