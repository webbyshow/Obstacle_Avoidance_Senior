import numpy as np
import cv2

def process_depth_image(depth_image, max_distance_mm):
    depth_filtered = np.where((depth_image > 0) & (depth_image <= max_distance_mm), depth_image, 0)
    depth_8bit = cv2.convertScaleAbs(depth_filtered, alpha=0.03)
    depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
    return depth_colormap, depth_8bit

def detect_contours(depth_8bit, obj_min_area):
    depth_binary = cv2.morphologyEx(depth_8bit, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > obj_min_area:
            x, y, w, h = cv2.boundingRect(contour)
            detected_contours.append((x, y, w, h))
    return detected_contours
