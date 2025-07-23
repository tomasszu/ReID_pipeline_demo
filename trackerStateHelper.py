from collections import defaultdict

class ReIDController:
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
        if current_zone == -1:
            return False  # skip tracks outside any zone
        info = self.track_info[track_id]
        return (info["zone"] != current_zone) or (info["id"] == -1)

    def update_state(self, track_id, zone, matched_id, similarity):
        info = self.track_info[track_id]
        info["zone"] = zone
        info["id"] = matched_id
        info["similarity"] = similarity

        if matched_id == -1:
            info["force_retry"] = True  # still needs retry
        else:
            info["force_retry"] = False  # success, no retry

    def get_current_id(self, track_id):
        return self.track_info[track_id]["id"]

    def get_similarity(self, track_id):
        return self.track_info[track_id]["similarity"]

    def match(self, crops):
        """
        Args:
            crops: list of (track_id, crop image)
        Only reIDs unknowns (id == -1) inside a valid zone.
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
        return {tid: info["id"] for tid, info in self.track_info.items()}

