import cv2

class CropZoneFilter:
    """A class to filter and crop detections based on defined zones in a video frame.
    This class allows for the definition of multiple zones in a video frame and filters detections based on their location within these zones.
    If enabled it acts as a filter to only accept detections that fall within the defined zones. This reduces noise and focuses on specific areas of interest.
    A vehicle is saved (for feature extraction) once every time it enters a new crop zone. The crop zones are defined by the user and can be adjusted based on the video frame size.
    The class can also to visualizes the zones and the center points of the detections.
    After filtering it crops the detected objects from the frame and stores them for further processing.
    A Vehicle Will never be saved fro cropping out if it has never entered a single defined crop zone. Alternatively, if the zone logic is disabled, all detections are cropped regardless of their location in every single frame anew.
    Attributes:
        rows (int): Number of rows to divide the frame into zones.
        cols (int): Number of columns to divide the frame into zones.
        area_bottom_left (tuple): Coordinates of the bottom-left corner of the area to crop (x_min, y_max).
        area_top_right (tuple): Coordinates of the top-right corner of the area to crop (x_max, y_min).
        debug (bool): If True, enables debug drawing on the frame.
        _pending_crops (list): List to hold cropped images for the current frame.
        zone_of_detections (dict): Dictionary to track which zone each detection belongs to.
    Methods:
        __init__(rows, cols, area_bottom_left, area_top_right, debug=True):
            Initializes the CropZoneFilter with the specified parameters.
        _generate_zones():
            Generates the zones based on the specified rows and columns within the defined area.
        _zone_of_point(point):
            Determines the zone index for a given point in the frame.
        _draw_debug(frame, center_points):
            Draws the zones and center points on the frame for debugging purposes.
        filter_and_crop(frame, detections, current_ids=None):
            Filters detections based on their location within the defined zones and crops the detected objects from the frame.
        get_crops():
            Returns the list of cropped images for the current frame.
    """
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
        """Generates the zones based on the specified rows and columns within the defined area.
        Returns:
            list: A list of tuples representing the zones, each defined by its top-left and bottom-right coordinates.
        """
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
        """Determines the zone index for a given point in the frame.
        Args:
            point (tuple): A tuple representing the coordinates of the point (x, y).
        Returns:
            int: The index of the zone that contains the point, or -1 if the point is not within any zone.
        """
        x, y = point
        for idx, (x1, y1, x2, y2) in enumerate(self.zones):
            if x1 <= x <= x2 and y1 <= y <= y2:
                return idx
        return -1

    def _draw_debug(self, frame, center_points):
        """Draws the zones and center points on the frame for debugging purposes.
        Args:
            frame (np.ndarray): The frame on which to draw the debug information.
            center_points (list): List of center points of the detections to be drawn.
        """
        for zone in self.zones:
            x1, y1, x2, y2 = zone
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
        for center in center_points:
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

    def filter_and_crop(self, frame, detections, current_ids=None):
        """Filters detections based on their location within the defined zones and crops the detected objects from the frame.
        It checks if the detection is within a zone and crops it accordingly.
        Args:
            frame (np.ndarray): The input frame from which to crop the detections.
            detections (sv.Detections): The detected vehicles in the frame, with tracking information.
            current_ids (dict, optional): A dictionary mapping object IDs to their current tracking IDs.
                If None, it will not filter based on current IDs.
        Returns:
            list: A list of filtered detections that are within the defined zones.
        """
        if current_ids is None:
            current_ids = {}
        
        self._pending_crops.clear()
        filtered = []
        center_points = []

        height, width = frame.shape[:2]
        

        for det in detections:
            # Clamp and fix bbox coordinates
            x1, y1, x2, y2 = map(int, det[5]["original_xyxy"])
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
                    should_crop = False

                    # New zone entry → definitely crop
                    if obj_id not in self.zone_of_detections or self.zone_of_detections[obj_id] != zone:
                        self.zone_of_detections[obj_id] = zone
                        should_crop = True

                    # If still not re-identified, keep retrying every frame
                    elif current_ids.get(obj_id, -1) == -1:
                        should_crop = True

                    if should_crop:
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
        """Returns the list of cropped images for the current frame.
        Returns:
            list: A list of tuples where each tuple contains the object ID and the corresponding cropped image.
        """
        return self._pending_crops
