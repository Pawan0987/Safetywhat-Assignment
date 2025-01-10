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
                image_path = retrieve_sub_object(frame, det, sub_object, "output")
                sub_object['image_path'] = image_path

    end_time = cv2.getTickCount()
    total_time = (end_time - start_time) / cv2.getTickFrequency()

    video.release()

    # Generate metadata
    metadata = generate_metadata(frame_count, total_time, min_fps, max_fps)

    # Combine frame detections and metadata
    output = {
        "frame_detections": all_detections,
        "metadata": metadata
    }

    # Save JSON output
    with open("output/object_detection_output.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"JSON output saved to 'output/object_detection_output.json'")

if __name__ == "__main__":
    main()
