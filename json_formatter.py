from datetime import datetime, timedelta
import psutil

def format_detections(detections, frame_id):
    formatted = {
        "frame_id": frame_id,
        "timestamp": (datetime.now() + timedelta(seconds=frame_id/30)).isoformat(),  
        "location": "Mumbai, Maharashtra, India", 
        "detections": []
    }

    for det in detections:
        formatted_det = {
            "object": det["object"],
            "id": det["id"],
            "bbox": det["bbox"],
            "confidence": det["confidence"]
        }
        if "subobject" in det:
            formatted_det["subobject"] = {
                "object": det["subobject"]["object"],
                "id": det["subobject"]["id"],
                "bbox": det["subobject"]["bbox"],
                "confidence": det["subobject"]["confidence"]
            }
            if "image_path" in det["subobject"]:
                formatted_det["subobject"]["image_path"] = det["subobject"]["image_path"]
        formatted["detections"].append(formatted_det)

    return formatted

def generate_metadata(total_frames, total_time, min_fps, max_fps):
    return {
        "processing_info": {
            "model_name": "YOLOv5s-Indian",
            "model_version": "1.0.0",
            "inference_device": "CPU",
            "average_fps": total_frames / total_time,
            "total_frames_processed": total_frames,
            "detection_threshold": 0.5,
            "processing_resolution": [1920, 1080]
        },
        "object_classes": {
            "main_objects": ["person", "car", "motorcycle", "auto-rickshaw"],
            "sub_objects": ["helmet", "tire", "license_plate"],
            "supported_pairs": [
                {"main": "person", "sub": "helmet"},
                {"main": "car", "sub": "tire"},
                {"main": "car", "sub": "license_plate"},
                {"main": "motorcycle", "sub": "license_plate"},
                {"main": "auto-rickshaw", "sub": "license_plate"}
            ]
        },
        "performance_metrics": {
            "average_inference_time": total_time / total_frames,
            "min_fps": min_fps,
            "max_fps": max_fps,
            "cpu_utilization": f"{psutil.cpu_percent()}%",
            "memory_usage": f"{psutil.virtual_memory().used / (1024 ** 3):.1f}GB"
        },
        "session_info": {
            "start_time": datetime.now().isoformat(),
            "end_time": (datetime.now() + timedelta(seconds=total_time)).isoformat(),
            "video_source": "mumbai_traffic.mp4"
        },
        "sub_object_retrieval": {
            "enabled": True,
            "save_path": "output/",
            "naming_convention": "{main_object}_{main_id}_{sub_object}_{sub_id}.jpg",
            "extraction_method": "crop_from_main_object"
        }
    }

