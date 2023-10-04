
start_time = 0
started_tracking = False
elapsed_time  = 0

distance_x, distance_y,distance_z = 0, 0, 0
angle = 0
apriltag_id = 'N/A'
closest_color_name = 'N/A'
average_color = (0, 0, 0)
servo_angle = 0

occlusion = 0

def initialize_servo():
    servo1.ChangeDutyCycle(0)
    return servo1, duty

# Initialize the servo
servo1, duty = initialize_servo()
pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model,xyz=args.xyz)
renderer = BlazeposeRenderer(pose, output=args.output)
detector = apriltag.Detector()

draw_xyz = False
while True:
    frame, body, occlusion = pose.next_frame(occlusion, started_tracking)
    if frame is None:
        break
    right_shoulder = None
    left_shoulder = None
    left_hip = None
    right_hip = None
    if body:
        print(f"Landmark score = {body.lm_score}")
        if started_tracking:
            elapsed_time = time.time() - start_time
            print(f"time: {elapsed_time}")
        right_shoulder = body.landmarks[KEYPOINT_DICT['right_shoulder'], :2]
        left_shoulder = body.landmarks[KEYPOINT_DICT['left_shoulder'], :2]
        left_hip = body.landmarks[KEYPOINT_DICT['left_hip'], :2]
        right_hip = body.landmarks[KEYPOINT_DICT['right_hip'], :2]

        points = [tuple(right_shoulder.astype(int)) if right_shoulder is not None else (0, 0),
                  tuple(left_shoulder.astype(int)) if left_shoulder is not None else (0, 0),
                  tuple(left_hip.astype(int)) if left_hip is not None else (0, 0),
                  tuple(right_hip.astype(int)) if right_hip is not None else (0, 0)]
        if all([right_shoulder is not None, left_shoulder is not None, 
                left_hip is not None, right_hip is not None]):
            top_left = tuple(right_shoulder.astype(int))
            bottom_right = (left_shoulder[0], left_hip[1])
            if right_shoulder[0] > left_shoulder[0]:
                print("User is facing away from the camera")
                top_left = tuple(left_shoulder.astype(int))
                bottom_right = (int(right_shoulder[0]), int(right_hip[1]))
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            if roi.size > 0:
                gray_cropped_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                pixels = roi.reshape(-1, 3)
                average_color = np.uint8(pixels).mean(axis=0)
                average_color = (average_color[2], average_color[1], average_color[0])
                if occlusion >  0 or not started_tracking:
                    if not started_tracking:
                        print("Please show your tag for 5 seconds to begin")
                    if started_tracking:
                        print("There is occlusion please show tag")
                    detected_tags = detector.detect(gray_cropped_img)
                    if len(detected_tags) == 0:
                        print("No AprilTag on body")
                        draw_xyz = False
                    else:
                        tag_ids = [detection.tag_id for detection in detected_tags]
                        if 187 in tag_ids:
                            print("Go on, that is the right body")
                            if not started_tracking:
                                start_time = time.time()
                                started_tracking = True
                                print("Tracking! thank you")
                            occlusion = 0
                            apriltag_id = 'A'
                        else:
                            print("There is an AprilTag, but not 187")
                            apriltag_id = 'N/A'
                            distance_x = 0
                            distance_y = 0
                            distance_z = 0
                            angle = 0
                if occlusion < 1 and started_tracking:
                    draw_xyz = True                   
                    try:
                        print(f"Distance: X:{body.xyz[0]/10:3.0f} cm")
                        print(f"Y:{body.xyz[1]/10:3.0f} cm")
                        print(f"Z:{body.xyz[2]/10:3.0f} cm")
                        distance_x = round(body.xyz[0] / 10, 3)
                        distance_y = round(body.xyz[1] / 10, 3)
                        distance_z = round(body.xyz[2] / 10, 3)
                        angle = degrees(atan2(body.xyz[0], body.xyz[2]))
                        print(f"angle: {angle} ")
                    except AttributeError:
                        pass 
                    angle_thresholds = [-45, -30, -15, 15, 30, 45]
                    duty_delta = 0.75
                    old_duty = duty
                def rgb_to_name(rgb_color):
                    min_color_diff = float("inf")
                    closest_color = None
                    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
                        rgb = webcolors.hex_to_rgb(key)
                        color_diff = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_color, rgb))
                        if color_diff < min_color_diff:
                            min_color_diff = color_diff
                            closest_color = name
                    return closest_color
                closest_color_name = rgb_to_name(average_color)
                message = f"{closest_color_name} shirt"
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(message, font, 1, 2)[0]
                text_x = (roi.shape[1] - text_size[0]) // 2
                text_y = (roi.shape[0] + text_size[1]) // 2
            else:
                print("Invalid or Empty ROI")
    frame = renderer.draw(frame, body, draw_xyz)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        servo1.ChangeDutyCycle(7)
        time.sleep(0.5)
        servo1.stop()
        GPIO.cleanup()
        print("Goodbye!")
        break

renderer.exit()
pose.exit()
