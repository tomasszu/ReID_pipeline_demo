from collections import defaultdict

class ReIDController:
    """Controller for managing ReID state and operations for vehicle tracking.
    This class handles the interaction between the database, feature extractor, and crop filter.
    It tracks the state of each vehicle, including which crop zone it is in, its current ID, and whether it needs to be re-identified.
    Attributes:
        db (LanceDBOperator): The database operator for managing vehicle features.
        extractor (ExtractingFeatures): The feature extractor for processing vehicle images.
        crop_filter (CropZoneFilter): The crop zone filter for managing vehicle crops and zones.
        track_info (defaultdict): A dictionary mapping track IDs to their state, including zone,
            current ID, whether it needs to be re-identified, and similarity score.
    """
    def __init__(self, db, extractor, crop_filter):
        self.db = db
        self.extractor = extractor
        self.crop_filter = crop_filter

        # track_id → dict with zone, id, force_retry, similarity
        self.track_info = defaultdict(lambda: {
            "zone": None,
            "id": -1,
            "force_retry": True,
            "similarity": 0.0
        })

    def should_reid(self, track_id, current_zone):
        """Determines if a vehicle should be re-identified based on its current zone and state.
        Args:
            track_id (int): The ID of the tracked vehicle.
            current_zone (int): The current zone of the vehicle.
        Returns:
            bool: True if the vehicle should be re-identified, False otherwise.
        """
        if current_zone == -1:
            return False  # skip tracks outside any zone
        info = self.track_info[track_id]
        return (info["zone"] != current_zone) or (info["id"] == -1)

    def update_state(self, track_id, zone, matched_id, similarity):
        """Updates the state of a tracked vehicle with the new zone, matched ID, and similarity score.
        Args:
            track_id (int): The ID of the tracked vehicle.
            zone (int): The zone in which the vehicle is currently located.
            matched_id (int): The ID of the vehicle as matched in the database, or -1 if not matched.
            similarity (float): The similarity score of the match.
        """
        info = self.track_info[track_id]
        info["zone"] = zone
        info["id"] = matched_id
        info["similarity"] = similarity

        if matched_id == -1:
            info["force_retry"] = True  # still needs retry
        else:
            info["force_retry"] = False  # success, no retry

    def get_current_id(self, track_id):
        """Retrieves the current ID of a tracked (tracked ID is kep as is for the first video - in the second video the vehicle is ReID'd from the first and the track id is replaced with the one from the first video) vehicle.
        Args:
            track_id (int): The ID of the tracked vehicle.
        Returns:
            int: The current ID of the vehicle, or -1 if not matched.
        """
        return self.track_info[track_id]["id"]

    def get_similarity(self, track_id):
        """Retrieves the similarity score of a tracked vehicle. 
        Args:
            track_id (int): The ID of the tracked vehicle.
        Returns:
            float: The similarity score of the vehicle, or 0.0 if not matched.
        """
        return self.track_info[track_id]["similarity"]

    def match(self, crops):
        """Processes a list of cropped images to extract features and query the database for matches.
        Args:
            crops (list): A list of tuples where each tuple contains the object ID and the corresponding cropped image.
        """
        for track_id, crop in crops:
            zone = self.crop_filter.zone_of_detections.get(track_id, -1)

            if not self.should_reid(track_id, zone):
                continue

            # proceed with feature extraction and DB query
            features = self.extractor.get_features_batch([(track_id, crop)])
            if features:
                result = self.db.query_features(features)
                if result:
                    matched_id = result[0][1]
                    dist = result[0][2]
                    similarity = 1.0 - dist
                else:
                    matched_id = -1
                    similarity = 0.0
            else:
                matched_id = -1
                similarity = 0.0

            self.update_state(track_id, zone, matched_id, similarity)

    def apply_to_detections(self, detections):
        """
        Overwrites detection tracker IDs and confidence using stored state.
        """
        for i, track_id in enumerate(detections.tracker_id):
            reid_id = self.get_current_id(track_id)
            similarity = self.get_similarity(track_id)

            detections.tracker_id[i] = reid_id
            detections.confidence[i] = similarity

    def get_all_ids(self):
        """Retrieves a mapping of all tracked IDs to their current IDs.
        Returns:
            dict: A dictionary mapping track IDs to their current IDs.
        """
        return {tid: info["id"] for tid, info in self.track_info.items()}

