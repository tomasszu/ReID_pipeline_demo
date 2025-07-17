import supervision as sv
import cv2

class Visualizer:
    def __init__(self, class_names: dict):
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.class_names = class_names

    def annotate(self, frame, detections):
        labels = [
            f"#{tracker_id} {self.class_names[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id, _
            in detections
        ]

        frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        return frame
