from detector import VehicleDetector
from visualizer import Visualizer
from cropZoneFilter import CropZoneFilter
from featureExtractor import ExtractingFeatures
from lanceDBOperator import LanceDBOperator
from trackerStateHelper import ReIDController

import cv2

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path1', type=str, default='videos/vdo4.avi', help='Path to the first video file. (Re-Identification FROM)')
    parser.add_argument('--video_path2', type=str, default='videos/vdo1.avi', help='Path to the second video file. (Re-Identification TO)')
    parser.add_argument('--roi_path1', type=str, default="videos/vdo4_roi.png", help='Path to the ROI image for the first video. If not provided, it will try to auto-detect in the same folder based on the video name.')
    parser.add_argument('--roi_path2', type=str, default="videos/vdo1_roi.png", help='Path to the ROI image for the second video. If not provided, it will try to auto-detect in the same folder based on the video name.')
    parser.add_argument('--detection_model_path', type=str, default='yolov8x.pt', choices=['yolov8x.pt', 'yolov8l.pt', 'yolov5su.pt'] , help='Path to the YOLO model file.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'], help='Device to run the model on (e.g., "cuda" or "cpu").')
    parser.add_argument('--start_offset_frames', type=int, default=0, help='Number of frames to delay processing at the start of the first video.')
    parser.add_argument('--reid_db_path', type=str, default='lancedb', help='Path to the LanceDB database for storing features.')
    parser.add_argument('--features_size', type=int, default=256, help='Size of the feature embeddings to be stored in the database.')
    return parser.parse_args()

def run_demo(video_path1, video_path2, roi_path1, roi_path2, detection_model, device, start_offset_frames):
    detector1 = VehicleDetector(video_path1, roi_path=roi_path1, model_path=detection_model)
    detector2 = VehicleDetector(video_path2, roi_path=roi_path2, model_path=detection_model)

    # Šis jāieliek kā arguments!!
    original_fps = detector2.cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {original_fps}")

    frame_shape = detector1.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), detector1.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_shape2 = detector2.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), detector2.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # Initialize crop zone filter with the frame shape
    crop_filter1 = CropZoneFilter(rows=7, cols=6, area_bottom_left= (0, 1000), area_top_right=(1750, 320), debug=True)
    crop_filter2 = CropZoneFilter(rows=7, cols=6, area_bottom_left= (200, 900), area_top_right=(1750, 320), debug=True)

    feature_extractor = ExtractingFeatures(device=device)

    db = LanceDBOperator("lancedb", features_size=256)

    reid_controller = ReIDController(db=db, extractor=feature_extractor, crop_filter=crop_filter2)


    visualizer1 = Visualizer(detector1.class_names)
    visualizer2 = Visualizer(detector2.class_names, traces=False)

    while True:
        ret1, frame1 = detector1.read_frame()
        ret2, frame2 = detector2.read_frame()

        if not ret1 or not ret2:
            print("End of video stream.")
            break

        # getting original and tracked detections
        detections1, frame1 = detector1.process_frame(frame1)
        detections2, frame2 = detector2.process_frame(frame2)

        # #Modified orifinals
        # print(f"Frame 1 detections: {orig_dets1}")
        # print(f"Frame 2 detections: {orig_dets2}")
        # # tracked detections
        # print(f"Tracked Frame 1 detections: {detections1}")
        # print(f"Tracked Frame 2 detections: {detections2}")
        


        # # Assign tracker IDs to original bboxes
        # transfer_tracker_ids(detections1, orig_dets1)
        # transfer_tracker_ids(detections2, orig_dets2)

        current_ids = reid_controller.get_all_ids()


        filtered_dets1 = crop_filter1.filter_and_crop(frame1, detections1, current_ids)
        filtered_dets2 = crop_filter2.filter_and_crop(frame2, detections2, current_ids)


        crops1 = crop_filter1.get_crops()
        crops2 = crop_filter2.get_crops()

        frame_id = detector1.get_current_frame_index()

        features1 = feature_extractor.get_features_batch(crops1)
        db.add_features(features1, frame_id)
        

        db.expire_old_features(frame_id, max_age=100)  # Keep features for X frames


        results = reid_controller.match(crops2)
        reid_controller.apply_to_detections(detections2)


        
        vis_frame1 = visualizer1.annotate(frame1, detections1)
        vis_frame2 = visualizer2.annotate(frame2, detections2)

        combined = cv2.hconcat([vis_frame1, vis_frame2])
        combined = cv2.resize(combined, (2560, 720))

        #vis_frame1 = cv2.resize(vis_frame1, (1280, 720))

        cv2.imshow("Vehicle Re-ID Demo", combined)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    detector1.release()
    detector2.release()
    cv2.destroyAllWindows()

    db.delete_table()  # Clean up the database table after demo

if __name__ == "__main__":
    #run_demo("video1.avi", "video2.avi")
    args = parse_args()

    run_demo(video_path1=args.video_path1, video_path2=args.video_path2, roi_path1=args.roi_path1, roi_path2=args.roi_path2, detection_model=args.detection_model_path, device=args.device, start_offset_frames=args.start_offset_frames)
