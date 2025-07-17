from detector import VehicleDetector
from visualizer import Visualizer
from cropZoneFilter import CropZoneFilter
from featureExtractor import ExtractingFeatures
from lanceDBOperator import LanceDBOperator
import cv2
import supervision as sv

def match_detections(detections: sv.Detections, reid_results):
    """
    Overwrite tracker IDs with ReID-matched IDs,
    and confidence scores with similarity (1 - distance).
    """
    reid_map = {obj_id: (matched_id, dist) for obj_id, matched_id, dist in reid_results}

    for i, track_id in enumerate(detections.tracker_id):
        if track_id in reid_map:
            matched_id, dist = reid_map[track_id]
            detections.tracker_id[i] = matched_id  # overwrite track ID
            detections.confidence[i] = 1 - dist    # similarity as confidence
        else:
            detections.tracker_id[i] = -1  # no match found
            detections.confidence[i] = 0.0


def run_demo(video_path1, video_path2, roi_path1=None):
    detector1 = VehicleDetector(video_path1, roi_path=roi_path1)
    detector2 = VehicleDetector(video_path2)

    frame_shape = detector1.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), detector1.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_shape2 = detector2.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), detector2.cap.get(cv2.CAP_PROP_FRAME_WIDTH)


    # Initialize crop zone filter with the frame shape
    crop_filter1 = CropZoneFilter(rows=7, cols=6, area_bottom_left= (0, 1000), area_top_right=(1750, 320), debug=True)
    crop_filter2 = CropZoneFilter()

    feature_extractor = ExtractingFeatures()

    db = LanceDBOperator("lancedb", features_size=256)


    visualizer1 = Visualizer(detector1.class_names)
    visualizer2 = Visualizer(detector2.class_names, traces=False)

    while True:
        ret1, frame1 = detector1.read_frame()
        ret2, frame2 = detector2.read_frame()

        if not ret1 or not ret2:
            print("End of video stream.")
            break

        detections1, frame1 = detector1.process_frame(frame1)
        detections2, frame2 = detector2.process_frame(frame2)

        # This returns only the new zone entries
        filtered_dets1 = crop_filter1.filter_and_crop(frame1, detections1)
        filtered_dets2 = crop_filter2.filter_and_crop(frame2, detections2)

        crops1 = crop_filter1.get_crops()
        crops2 = crop_filter2.get_crops()

        features1 = feature_extractor.get_features_batch(crops1)
        features2 = feature_extractor.get_features_batch(crops2)

        db.add_features(features1)
        results = db.query_features(features2)

        match_detections(detections2, results)

        
        vis_frame1 = visualizer1.annotate(frame1, detections1)
        vis_frame2 = visualizer2.annotate(frame2, detections2)

        combined = cv2.hconcat([vis_frame1, vis_frame2])
        combined = cv2.resize(combined, (2560, 720))

        #vis_frame1 = cv2.resize(vis_frame1, (1280, 720))

        cv2.imshow("Vehicle Re-ID Demo", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector1.release()
    detector2.release()
    cv2.destroyAllWindows()

    db.delete_table()  # Clean up the database table after demo

if __name__ == "__main__":
    #run_demo("video1.avi", "video2.avi")
    run_demo(video_path1="videos/vdo4.avi", video_path2="videos/vdo1.avi", roi_path1="videos/vdo4_roi.png")
