import cv2
import json
from detector import IndianObjectDetector
from json_formatter import format_detections, generate_metadata
from sub_object_retriever import retrieve_sub_object
from config import Config

def main():
    # Initialize the detector
    detector = IndianObjectDetector(Config.YOLO_WEIGHTS, Config.YOLO_CONFIG)

    # Open the video file
    video = cv2.VideoCapture(Config.VIDEO_PATH)

    frame_count = 0
    total_time = 0
    all_detections = []
    min_fps = float('inf')
    max_fps = 0

    start_time = cv2.getTickCount()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # Perform detection
        frame_start_time = cv2.getTickCount()
        detections = detector.detect(frame)
        frame_end_time = cv2.getTickCount()

        # Calculate processing time and FPS
        process_time = (frame_end_time - frame_start_time) / cv2.getTickFrequency()
        fps = 1 / process_time

        total_time += process_time
        frame_count += 1

        min_fps = min(min_fps, fps)
        max_fps = max(max_fps, fps)

        # Format detections to JSON
        formatted_detections = format_detections(detections, frame_count)
        all_detections.append(formatted_detections)

        # Retrieve and save sub-object images
        for det in detections:
            if 'subobject' in det:
                sub_object = det['subobject']
                image_path = retrieve_sub_object(frame, det, sub_object, Config.OUTPUT_PATH)
                sub_object['image_path'] = image_path

        # Display the results (optional for demo purposes)
        for det in detections:
            cv2.rectangle(frame, (det['bbox'][0], det['bbox'][1]), (det['bbox'][2], det['bbox'][3]), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['object']} {det['id']}", (det['bbox'][0], det['bbox'][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Indian Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end_time = cv2.getTickCount()
    total_time = (end_time - start_time) / cv2.getTickFrequency()

    video.release()
    cv2.destroyAllWindows()

    # Generate metadata
    metadata = generate_metadata(frame_count, total_time, min_fps, max_fps)

    # Combine frame detections and metadata
    output = {
        "frame_detections": all_detections,
        "metadata": metadata
    }

    # Save JSON output
    with open(f"{Config.OUTPUT_PATH}/indian_object_detection_output.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"JSON output saved to '{Config.OUTPUT_PATH}/indian_object_detection_output.json'")

if __name__ == "__main__":
    main()

