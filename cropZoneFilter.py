import cv2

class CropZoneFilter:
    def __init__(self, rows=None, cols=None,
                 area_bottom_left = None, area_top_right=None, debug=True):
        self.rows = rows
        self.cols = cols
        self.debug = debug
        self._pending_crops = []  # holds (id, crop) for the current frame
        self.zone_of_detections = {}

        # Set the cropping area
        self.area_bottom_left = area_bottom_left  # (x_min, y_max)
        self.area_top_right = area_top_right      # (x_max, y_min)

        # asserts that the points are of valid structure
        def _assert_point(name, point):
            assert isinstance(point, (tuple, list)), f"{name} must be a tuple or list"
            assert len(point) == 2, f"{name} must have exactly two elements (x, y)"
            assert all(isinstance(v, (int, float)) for v in point), f"{name} values must be int or float"

        if self.rows is not None and self.cols is not None \
        and self.area_bottom_left is not None and self.area_top_right is not None:

            _assert_point("area_bottom_left", self.area_bottom_left)
            _assert_point("area_top_right", self.area_top_right)

            self.zones = self._generate_zones()
            self.use_zones = True
        else:
            self.zones = []
            self.use_zones = False  # zone filtering disabled


    def _generate_zones(self):
        x_min, y_max = self.area_bottom_left
        x_max, y_min = self.area_top_right

        zone_width = (x_max - x_min) / self.cols
        zone_height = (y_max - y_min) / self.rows
        zones = []

        for i in range(self.rows):
            for j in range(self.cols):
                x1 = x_min + j * zone_width
                y1 = y_min + i * zone_height
                x2 = x_min + (j + 1) * zone_width
                y2 = y_min + (i + 1) * zone_height
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        for center in center_points:
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    def filter_and_crop(self, frame, detections):
        """
        Args:
            frame: current frame
            detections: list of tuples (xyxy, class_id, conf, cls_name, tracker_id)

        Returns:
            filtered_detections: detections that are accepted (based on zone logic or unconditionally)
        """
        self._pending_crops.clear()
        filtered = []
        center_points = []

        height, width = frame.shape[:2]

        for det in detections:
            # Clamp and fix bbox coordinates
            x1, y1, x2, y2 = map(int, det[0])
            x1 = max(1, min(x1, width - 2))
            y1 = max(1, min(y1, height - 2))
            x2 = max(x1 + 1, min(x2, width - 1))
            y2 = max(y1 + 1, min(y2, height - 1))

            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            center_points.append(center)

            det = ((x1, y1, x2, y2), *det[1:])  # Update bbox

            obj_id = det[4]  # tracker ID

            if self.use_zones:
                zone = self._zone_of_point(center)
                if zone != -1:
                    if obj_id not in self.zone_of_detections or self.zone_of_detections[obj_id] != zone:
                        self.zone_of_detections[obj_id] = zone
                        filtered.append(det)
                        crop = frame[y1:y2, x1:x2].copy()
                        if crop.size > 0:
                            self._pending_crops.append((obj_id, crop))
            else:
                # No zone logic: crop everything
                filtered.append(det)
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    self._pending_crops.append((obj_id, crop))

        if self.debug and self.use_zones:
            self._draw_debug(frame, center_points)

        return filtered

    def get_crops(self):
        return self._pending_crops
