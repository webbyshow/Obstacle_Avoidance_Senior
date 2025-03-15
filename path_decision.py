def make_decision(detected_objects, safe_distance, angle_stabilization, prev_turn_angle, img_width):
    command_text = "FORWARD"
    turn_angle = prev_turn_angle  # ตั้งค่าเริ่มต้นให้กับ turn_angle เป็น prev_turn_angle

    if detected_objects:
        closest_x, _, closest_depth, x1, y1, x2, y2, conf, cls = detected_objects[0]
        if closest_depth < safe_distance:
            turn_angle = int(((x2 - x1) / img_width) * 60)
            if prev_turn_angle is not None and abs(turn_angle - prev_turn_angle) < angle_stabilization:
                turn_angle = prev_turn_angle
            if closest_x < img_width * 0.5:
                command_text = f"TURN RIGHT {turn_angle} degrees"
            else:
                command_text = f"TURN LEFT {turn_angle} degrees"

    return command_text, turn_angle
