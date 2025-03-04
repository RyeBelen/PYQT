import cv2

from ultralytics import YOLO
import supervision as sv
import numpy as np

ZONE_POLYGON = np.array([
    [-0.25, 0.2],
    [0.25, 0.2],
    [0.25, -0.2],
    [-0.25, -0.2]
]) + np.array([0.3, 0.5])  # Centered polygon

vehicle_classes = {2,4,5,7}

def main():
    video_path = "testvids/test_footage_1.mp4"  # Set your video file path here
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model_path = "models/traffico_v1.pt"
    model = YOLO(model_path)

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5,
        color=sv.Color.white()
    )

    zone_polygon = (ZONE_POLYGON * np.array([frame_width, frame_height])).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=(frame_width, frame_height))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)

        filtered_detections = sv.Detections(
            xyxy=np.array([
                bbox for bbox, _, class_id, _ in detections if class_id in vehicle_classes
            ]),
            confidence=np.array([
                confidence for _, confidence, class_id, _ in detections if class_id in vehicle_classes
            ]),
            class_id=np.array([
                class_id for _, _, class_id, _ in detections if class_id in vehicle_classes
            ]),
            tracker_id=np.array([
                tracker_id for _, _, class_id, tracker_id in detections if class_id in vehicle_classes
            ])
        )

        # Convert bounding boxes to center points
        centers_xy = np.array([
            [(x_min + x_max) / 2, (y_min + y_max) / 2]  # (center_x, center_y)
            for x_min, y_min, x_max, y_max in filtered_detections.xyxy
        ])

        # Create a new sv.Detections object with centers instead of bounding boxes
        center_detections = sv.Detections(
            xyxy=np.hstack([centers_xy, centers_xy]),  # Duplicate centers to match xyxy format
            confidence=filtered_detections.confidence,
            class_id=filtered_detections.class_id,
            tracker_id=filtered_detections.tracker_id
        )

        # Annotate boxes and labels
        labels = [
            f"{model.names[class_id]} {confidence:0.9f}"
            for _, confidence, class_id, _ in filtered_detections
        ]
        frame = box_annotator.annotate(scene=frame, detections=filtered_detections, labels=labels)

        # Annotate center points
        for (cx, cy) in centers_xy:
            cv2.circle(frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)  # Green dot for center


        zone.trigger(detections=center_detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("YOLOv8 Video", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
