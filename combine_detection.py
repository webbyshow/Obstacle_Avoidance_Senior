import cv2
import numpy as np
def calculate_depth_in_roi(depth_image, bbox):
    x1, y1, w, h = bbox
    center_x, center_y = x1 + w // 2, y1 + h // 2
    roi_width, roi_height = int(w * 0.4), int(h * 0.4)
    roi_x, roi_y = center_x - roi_width // 2, center_y - roi_height // 2

    roi = depth_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    valid_depths = roi[roi > 0]  # Exclude zero values
    if valid_depths.size > 0:
        mean_depth = np.mean(valid_depths) / 1000  # Convert mm to meters if needed
    else:
        mean_depth = 0
    return mean_depth

def merge_detections(color_image, depth_colormap, detected_contours, detected_objects, depth_image, model):
    # Draw contours from depth map
    for x, y, w, h in detected_contours:
        # Make sure the coordinates are integer
        x, y, w, h = map(int, [x, y, w, h])
        cv2.rectangle(color_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(depth_colormap, (x, y), (x + w, y + h), (255, 255, 255), 2)
        mean_depth = calculate_depth_in_roi(depth_image, (x, y, w, h))
        depth_text = f"Depth: {mean_depth:.2f}m" if mean_depth > 0 else "Depth: No data"
        cv2.putText(color_image, depth_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Draw detections from YOLO
    for obj in detected_objects:
        x1, y1, x2, y2 = map(int, [obj[3], obj[4], obj[5], obj[6]])  # Convert to int
        conf, cls, depth = obj[7], obj[8], obj[2]
        label = f"{model.names[int(cls)]}: {conf:.2f}"
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        depth_text = f"Depth: {depth/1000:.2f}m" if depth > 0 else "Depth: No data"
        cv2.putText(color_image, depth_text, (x1, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
