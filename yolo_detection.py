from ultralytics import YOLO
import numpy as np

def load_yolo_model():
    model = YOLO("C:/Users/user/Downloads/best.pt")
    return model

def detect_objects(model, color_image, conf_threshold, depth_image):
    results = model(color_image)[0]
    detected_objects = []
    for box in results.boxes.data:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if conf < conf_threshold:
            continue
        # Ensure the center coordinates are integers for slicing
        center_x = int((x1 + x2) // 2)
        center_y = int((y1 + y2) // 2)

        # Define ROI safely within the bounds of the depth_image
        depth_roi = depth_image[max(0, center_y-2):min(center_y+3, depth_image.shape[0]),
                                max(0, center_x-2):min(center_x+3, depth_image.shape[1])]
        obj_depth = np.median(depth_roi[depth_roi > 0])

        if obj_depth > 0:
            detected_objects.append((center_x, center_y, obj_depth, x1, y1, x2, y2, conf, cls))
    
    detected_objects.sort(key=lambda obj: obj[2])
    
    return detected_objects
