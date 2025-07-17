import cv2

class CropZoneFilter:
    def __init__(self, frame_shape, rows=8, cols=4, y_threshold=325, debug=True):
        self.rows = rows
        self.cols = cols
        self.y_threshold = y_threshold
        self.zones = self._generate_zones(frame_shape)
        self.zone_of_detections = {}  # id -> last seen zone
        self.debug = debug
        self._pending_crops = []  # holds (id, crop) for the current frame

    def _generate_zones(self, frame_shape):
        h, w = frame_shape[:2]
        zone_width = w // self.cols
        zone_height = h // self.rows
        zones = []

        for i in range(self.rows):
            for j in range(self.cols):
                x1 = j * zone_width
                y1 = i * zone_height
                x2 = (j + 1) * zone_width
                y2 = (i + 1) * zone_height
                if y1 > self.y_threshold:
                    zones.append((int(x1), int(y1), int(x2), int(y2)))
        return zones

    def _zone_of_point(self, point):
        x, y = point
        for idx, (x1, y1, x2, y2) in enumerate(self.zones):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return -1

    def _draw_debug(self, frame, center_points):
        for zone in self.zones:
            x1, y1, x2, y2 = zone
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        for center in center_points:
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    def filter_and_crop(self, frame, detections):
        """
        Args:
            frame: current frame
            detections: list of tuples (xyxy, class_id, conf, cls_name, tracker_id)

        Returns:
            filtered_detections: list of detections (same format) that just entered a new zone
        """
        self._pending_crops.clear()
        filtered = []
        center_points = []

        height, width = frame.shape[:2]

        for det in detections:
            # Clamp coordinates before use
            x1, y1, x2, y2 = map(int, det[0])
            x1 = max(1, min(x1, width - 2))
            y1 = max(1, min(y1, height - 2))
            x2 = max(x1 + 1, min(x2, width - 1))   # x2 must be > x1
            y2 = max(y1 + 1, min(y2, height - 1))  # y2 must be > y1

            center = (int((x1 + x2) // 2), int((y1 + y2) // 2))
            center_points.append(center)

            # Store fixed coordinates back in detection (optional)
            det = ((x1, y1, x2, y2), *det[1:])

            zone = self._zone_of_point(center)
            if zone != -1:
                obj_id = det[4]  # tracker ID
                if obj_id not in self.zone_of_detections or self.zone_of_detections[obj_id] != zone:
                    self.zone_of_detections[obj_id] = zone
                    filtered.append(det)

                    # Crop and store
                    crop = frame[y1:y2, x1:x2].copy()
                    if crop.size > 0:
                        self._pending_crops.append((obj_id, crop))

        if self.debug:
            self._draw_debug(frame, center_points)

        return filtered

    def get_crops(self):
        """
        Returns:
            List of tuples (id, crop_image)
        """
        return self._pending_crops
