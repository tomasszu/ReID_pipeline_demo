import cv2
import supervision as sv
from ultralytics import YOLO
import numpy as np
import os
from supervision.detection.utils import box_iou_batch

class VehicleDetector:
    def __init__(self, model_path, device, video_path: str, class_ids=None, roi_path=None, start_offset_frames: int = 0):
        self.cap = cv2.VideoCapture(video_path)
        self.model = YOLO(model_path).to(device)
        self.tracker = sv.ByteTrack()
        self.device = device

        self.class_ids = class_ids if class_ids else [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.class_names = self.model.model.names
        self.conf_threshold = 0.6

        self.roi_mask = self._load_roi(roi_path, video_path)


        self.delay_frames = start_offset_frames
        self.frozen_frame = None
        self.current_frame_index = 0

    def _load_roi(self, roi_path, video_path):
        # If explicitly given, try to load
        if roi_path and os.path.exists(roi_path):
            print(f"Using provided ROI from: {roi_path}")
            return cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)

        # Otherwise try to derive from video path
        base_path, _ = os.path.splitext(video_path)
        auto_roi_path_a = base_path + "_roi.png"
        auto_roi_path_b = base_path + "roi.png"
        auto_roi_path_c = base_path + "-roi.png"
        auto_roi_path_d = base_path + "Roi.png"

        auto_roi_path = auto_roi_path_a if os.path.exists(auto_roi_path_a) else \
                        auto_roi_path_b if os.path.exists(auto_roi_path_b) else \
                        auto_roi_path_c if os.path.exists(auto_roi_path_c) else \
                        auto_roi_path_d if os.path.exists(auto_roi_path_d) else None

        if auto_roi_path:
            print(f"Using auto-detected ROI from: {auto_roi_path}")
            # Ensure the ROI is grayscale
            return cv2.imread(auto_roi_path, cv2.IMREAD_GRAYSCALE)

        print(f" No auto found ROI with filename {auto_roi_path}. Using full frame.")
        return None


    def read_frame(self):
        if self.delay_frames > 0:
            if self.frozen_frame is None:
                ret, frame = self.cap.read()
                if not ret:
                    return False, None
                self.frozen_frame = frame.copy()
            self.delay_frames -= 1
            return True, self.frozen_frame.copy()
        else:
            ret, frame = self.cap.read()
            return ret, frame

    def process_frame(self, frame):

        if self.roi_mask is not None:
            roi_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
        else:
            roi_frame = frame.copy()

        results = self.model(roi_frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Save original before tracking
        length = len(detections.xyxy)

        # Filter classes & confidence
        detections = detections[np.isin(detections.class_id, self.class_ids)]
        detections = detections[np.greater(detections.confidence, self.conf_threshold)]

        # print(f"Orig detections: {len(detections.xyxy)} ---------------------------------------------")
        # print(detections)

        # Update with tracker
        tracked_detections = self.tracker.update_with_detections(detections)

        if detections.xyxy.shape[0] > 0 and tracked_detections.xyxy.shape[0] > 0:

            # match tracked boxes with original detections using IoU
            iou_matrix = box_iou_batch(detections.xyxy, tracked_detections.xyxy)

            # Assign each tracked box to the detection with the highest IoU
            best_matches = iou_matrix.argmax(axis=0)  # shape: (num_tracked,)

            # Now for each tracked detection, attach the original box
            original_boxes = detections.xyxy[best_matches]

            # Attach to .data["original_xyxy"]
            tracked_detections.data["original_xyxy"] = original_boxes

        else:
            tracked_detections.data["original_xyxy"] = np.empty((0, 4), dtype=np.float32)
            # print("No detections or tracked detections found.")
            # print(f"Tracked detections: {len(tracked_detections.xyxy)} -----------------------------------")
            # print(detections)
            # print(f"Original detections: {length} -----------------------------------")
            # print(tracked_detections)

        # print(f"Tracked detections: {len(tracked_detections.xyxy)} -----------------------------------")
        # print(tracked_detections)


        return tracked_detections, frame

    def release(self):
        self.cap.release()

    def get_current_frame_index(self):
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
