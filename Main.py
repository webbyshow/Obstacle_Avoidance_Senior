import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import os
from ultralytics import YOLO
from config import setup_realsense
from record_video import setup_video_writers
from yolo_detection import load_yolo_model, detect_objects
from depth_map_detection import process_depth_image, detect_contours
from combine_detection import merge_detections
from path_decision import make_decision

def main():
    width = 640
    height = 480
    MAX_DISTANCE_M = 5.0  # Maximum distance to consider (In meter)
    MAX_DISTANCE_MM = int(MAX_DISTANCE_M * 1000)  # Convert meters to mm
    CONF_THRESHOLD = 0.05  # Minimum confidence for YOLO detections
    OBJ_MIN_AREA = (500 * width * height) / (640 * 480)  # Minimum area for depth-based contour detection
    SAFE_DISTANCE = 1500  # Distance in mm to be considered safe for moving forward
    ANGLE_STABILIZATION = 5  # Prevent angle fluctuation by allowing minor variations

    prev_turn_angle = None  # Initialize the previous turn angle

    pipeline = setup_realsense()
    #rgb_out, depth_out, rgb_no_label_out = setup_video_writers("recorded_videos")
    model = load_yolo_model()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            color_no_label = color_image.copy()  # Copy for no-label video

            depth_colormap, depth_8bit = process_depth_image(depth_image, MAX_DISTANCE_MM)
            detected_contours = detect_contours(depth_8bit, OBJ_MIN_AREA)
            detected_objects = detect_objects(model, color_image, CONF_THRESHOLD,depth_image)

            merge_detections(color_image, depth_colormap, detected_contours, detected_objects, depth_image,model)
            #ลองหา make_decision2,3,4 มาเปลี่ยนได้
            command_text, prev_turn_angle = make_decision(detected_objects, SAFE_DISTANCE, ANGLE_STABILIZATION, prev_turn_angle, width)

            cv2.putText(color_image, command_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            #rgb_out.write(color_image)
            #depth_out.write(depth_colormap)
            #rgb_no_label_out.write(color_no_label)

            cv2.imshow('RGB Camera (Obstacle Detection)', color_image)
            cv2.imshow('Depth Map', depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        pipeline.stop()
        #rgb_out.release()
        #depth_out.release()
        #rgb_no_label_out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
