import os
import cv2

def setup_video_writers(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    rgb_video_path = os.path.join(output_folder, "rgb_output.avi")
    depth_video_path = os.path.join(output_folder, "depth_output.avi")
    rgb_no_label_path = os.path.join(output_folder, "rgb_no_label.avi")

    rgb_out = cv2.VideoWriter(rgb_video_path, fourcc, 20.0, (640, 480))
    depth_out = cv2.VideoWriter(depth_video_path, fourcc, 20.0, (640, 480))
    rgb_no_label_out = cv2.VideoWriter(rgb_no_label_path, fourcc, 20.0, (640, 480))

    return rgb_out, depth_out, rgb_no_label_out
