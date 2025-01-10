import cv2
import torch
from ultralytics import YOLO

class IndianObjectDetector:
    def __init__(self, weights_path, config_path):
        self.model = YOLO(weights_path)
        self.object_id = 0
        self.sub_object_id = 0

    def detect(self, frame):
        results = self.model(frame)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                self.object_id += 1
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf.item()
                class_id = box.cls.item()
                class_name = self.model.names[int(class_id)]

                obj = {
                    "object": class_name,
                    "id": self.object_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(confidence)
                }

                # Detect sub-objects
                if obj["object"] == "person":
                    sub_obj = self.detect_sub_object(frame[int(y1):int(y2), int(x1):int(x2)], "helmet")
                    if sub_obj:
                        sub_obj["bbox"] = [int(x1 + sub_obj["bbox"][0]), int(y1 + sub_obj["bbox"][1]),
                                           int(x1 + sub_obj["bbox"][2]), int(y1 + sub_obj["bbox"][3])]
                        obj["subobject"] = sub_obj
                elif obj["object"] in ["car", "motorcycle", "auto-rickshaw"]:
                    sub_obj = self.detect_sub_object(frame[int(y1):int(y2), int(x1):int(x2)], "license_plate")
                    if sub_obj:
                        sub_obj["bbox"] = [int(x1 + sub_obj["bbox"][0]), int(y1 + sub_obj["bbox"][1]),
                                           int(x1 + sub_obj["bbox"][2]), int(y1 + sub_obj["bbox"][3])]
                        obj["subobject"] = sub_obj

                detections.append(obj)

        return detections

    def detect_sub_object(self, sub_frame, sub_object_class):
        # Simplified sub-object detection (you may want to use a separate model for this)
        self.sub_object_id += 1
        gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
        if sub_object_class == "helmet":
            # Simple circle detection for helmet
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
            if circles is not None:
                circles = circles[0, :][0]
                return {
                    "object": sub_object_class,
                    "id": self.sub_object_id,
                    "bbox": [int(circles[0] - circles[2]), int(circles[1] - circles[2]),
                             int(circles[0] + circles[2]), int(circles[1] + circles[2])],
                    "confidence": 0.8
                }
        elif sub_object_class == "license_plate":
            # Simple rectangle detection for license plate
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 2 < aspect_ratio < 5:  # Typical aspect ratio for license plates
                    return {
                        "object": sub_object_class,
                        "id": self.sub_object_id,
                        "bbox": [x, y, x + w, y + h],
                        "confidence": 0.75
                    }
        return None

