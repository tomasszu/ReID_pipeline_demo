import supervision as sv
import cv2

class Visualizer:
    def __init__(self, class_names: dict, traces=True):
        self.box_annotator = sv.BoundingBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.class_names = class_names
        self.draw_traces = traces

    def annotate(self, frame, detections):
        labels = []
        for _, _, confidence, class_id, tracker_id, _ in detections:
            if tracker_id == -1:
                label = "Unknown"
            else:
                name = self.class_names.get(class_id, "Vehicle")
                label = f"ID {tracker_id} {name} {confidence:.2f}"
            labels.append(label)

        frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        frame = self.label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        if self.draw_traces:
            frame = self.trace_annotator.annotate(scene=frame, detections=detections)

        return frame
