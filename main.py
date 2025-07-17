from detector import VehicleDetector
from visualizer import Visualizer
from cropZoneFilter import CropZoneFilter
from featureExtractor import ExtractingFeatures
from lanceDBOperator import LanceDBOperator
import cv2

def run_demo(video_path1, roi_path1=None):
    detector1 = VehicleDetector(video_path1, roi_path=roi_path1)
    #detector2 = VehicleDetector(video_path2)

    frame_shape = detector1.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), detector1.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    #frame_shape2 = detector2.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), detector2.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    crop_filter1 = CropZoneFilter(frame_shape, debug=True)

    feature_extractor = ExtractingFeatures()

    db = LanceDBOperator("lancedb", features_size=256)


    visualizer = Visualizer(detector1.class_names)  # Assuming same model

    while True:
        ret1, frame1 = detector1.read_frame()
        #ret2, frame2 = detector2.read_frame()

        # if not ret1 or not ret2:
        #     break

        if not ret1:
            break

        detections1, frame1 = detector1.process_frame(frame1)
        #detections2, frame2 = detector2.process_frame(frame2)

        print(detections1)
        # This returns only the new zone entries
        filtered_dets1 = crop_filter1.filter_and_crop(frame1, detections1)
        #filtered_dets2 = crop_filter2.filter_and_crop(frame2, detections2)

        crops1 = crop_filter1.get_crops()

        # for obj_id, crop in crops1:
        #     print(f"Object ID: {obj_id}, Crop shape: {crop.shape}")
        #     # # now you can pass it to the feature extractor
        #     # feature_extractor.process(crop, obj_id)

        features = feature_extractor.get_features_batch(crops1)

        db.add_features(features)

        #print(features)

        
        vis_frame1 = visualizer.annotate(frame1, detections1)
        #vis_frame2 = visualizer.annotate(frame2, detections2)

        #combined = cv2.hconcat([vis_frame1, vis_frame2])
        #combined = cv2.resize(combined, (1600, 600))

        vis_frame1 = cv2.resize(vis_frame1, (1280, 720))

        cv2.imshow("Vehicle Re-ID Demo", vis_frame1)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    detector1.release()
    #detector2.release()
    cv2.destroyAllWindows()

    db.delete_table()  # Clean up the database table after demo

if __name__ == "__main__":
    #run_demo("video1.avi", "video2.avi")
    run_demo("videos/vdo4.avi", roi_path1="videos/roi/vdo4_roi.png")
