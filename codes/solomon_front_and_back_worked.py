import cv2
from math import atan2, degrees
import sys
import numpy as np
import apriltag
import RPi.GPIO as GPIO
import time
import argparse
import webcolors
sys.path.append("../..")
from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='full',
                        help="Landmark model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
parser.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
args = parser.parse_args()  

def initialize_servo():
    GPIO.setmode(GPIO.BOARD)
    GPIO.cleanup()
    GPIO.setup(11, GPIO.OUT)
    servo1 = GPIO.PWM(11, 50) # pin 11 for servo1, pulse 50Hz
    servo1.start(0) # Start PWM running, with value of 0 (pulse off)
    time.sleep(0.5)
    duty = 7
    servo1.ChangeDutyCycle(duty)
    time.sleep(0.5)
    servo1.ChangeDutyCycle(0)
    return servo1, duty

# Initialize the servo
servo1, duty = initialize_servo()
pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model,xyz=args.xyz)
renderer = BlazeposeRenderer(pose, output=args.output)
detector = apriltag.Detector()

draw_xyz = False
while True:
    # Run blazepose on the nexray_framet frame
    frame, body = pose.next_frame()
    if frame is None:
        break

    # Get the coordinates of the shoulder and hip points
    right_shoulder = None
    left_shoulder = None
    left_hip = None
    right_hip = None
    if body:
        right_shoulder = body.landmarks[KEYPOINT_DICT['right_shoulder'], :2]
        left_shoulder = body.landmarks[KEYPOINT_DICT['left_shoulder'], :2]
        left_hip = body.landmarks[KEYPOINT_DICT['left_hip'], :2]
        right_hip = body.landmarks[KEYPOINT_DICT['right_hip'], :2]

        # Define the four points of the quadrilateral
        points = [tuple(right_shoulder.astype(int)) if right_shoulder is not None else (0, 0),
                  tuple(left_shoulder.astype(int)) if left_shoulder is not None else (0, 0),
                  tuple(left_hip.astype(int)) if left_hip is not None else (0, 0),
                  tuple(right_hip.astype(int)) if right_hip is not None else (0, 0)]
        
        # the points
        # print(f"Debugging Landmarks: right_shoulder = {points[0]}, left_shoulder = {points[1]}, left_hip = {points[2]}, right_hip = {points[3]}")


        if all([right_shoulder is not None, left_shoulder is not None, 
                left_hip is not None, right_hip is not None]):

            # Draw the quadrilateral (green color)
            #cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Determine the top-left corner of the rectangle (right shoulder)
            top_left = tuple(right_shoulder.astype(int))

            # Determine the bottom-right corner of the rectangle
            bottom_right = (left_shoulder[0], left_hip[1])

            # Check if the user is facing away from the camera
            if right_shoulder[0] > left_shoulder[0]:
                print("User is facing away from the camera")
                # In this case, use left shoulder as the top-left corner and get the bottom-right accordingly
                top_left = tuple(left_shoulder.astype(int))
                bottom_right = (int(right_shoulder[0]), int(right_hip[1]))

            cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), thickness=3)

            # Print coordinate Before defining the ROI for debug
            # print(f"Debugging Coordinates: top_left = {top_left}, bottom_right = {bottom_right}, frame shape = {frame.shape}")


            # Crop the ROI within the rectangle
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # # Determine the minimum and maximum x, y coordinates from the keypoints
            # min_x = int(min(right_shoulder[0], left_shoulder[0], right_hip[0], left_hip[0]))
            # min_y = int(min(right_shoulder[1], left_shoulder[1], right_hip[1], left_hip[1]))
            # max_x = int(max(right_shoulder[0], left_shoulder[0], right_hip[0], left_hip[0]))
            # max_y = int(max(right_shoulder[1], left_shoulder[1], right_hip[1], left_hip[1]))

            # # Define the top-left and bottom-right corners of your rectangle
            # top_left = (min_x, min_y)
            # bottom_right = (max_x, max_y)

            # cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), thickness=3)
            # # Crop the ROI within the rectangle
            # roi = frame[min_y:max_y, min_x:max_x]            
            # Check if the ROI is empty or invalid
            if roi.size > 0:
                gray_cropped_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # Reshape the ROI into an array of pixel values
                pixels = roi.reshape(-1, 3)
                # Calculate the mode (most frequent color) in the ROI
                dominant_color = np.uint8(pixels).mean(axis=0)

                # Convert the dominant color to a tuple (B, G, R)
                dominant_color = (dominant_color[2], dominant_color[1], dominant_color[0])

                # Detect AprilTag in the cropped image
        
                detected_tags = detector.detect(gray_cropped_img)
                
                if len(detected_tags) == 0:
                    print("No AprilTag on body")
                    draw_xyz = False
                else:
                    #print(detected_tags)    
                    tag_ids = [detection.tag_id for detection in detected_tags]
                    if 187 in tag_ids:
                        print("Go on, that is the right body")
                        draw_xyz = True
                        
                        print(f"Distance: X:{body.xyz[0]/10:3.0f} cm") 
                        print(f"Y:{body.xyz[1]/10:3.0f} cm")
                        print(f"Z:{body.xyz[2]/10:3.0f} cm")

                        angle = degrees(atan2(body.xyz[0], body.xyz[2]))
                        print(f"angle: {angle} ")

                        # Define angle thresholds and the duty increment/decrement value
                        angle_thresholds = [-60, -40, -20, 20, 40, 60]
                        duty_delta = 1

                        # Initialize old_duty to track previous duty value
                        old_duty = duty

                        # Handle negative angles
                        if angle <= -20 and duty < 11:
                            for threshold in reversed(angle_thresholds[:3]):
                                if angle < threshold:
                                    duty += duty_delta

                        # Handle positive angles
                        elif angle >= 20 and duty > 3:
                            for threshold in angle_thresholds[3:]:
                                if angle > threshold:
                                    duty -= duty_delta

                        # Update the servo
                        if old_duty != duty:
                            servo1.ChangeDutyCycle(duty)
                            time.sleep(0.5)
                            servo1.ChangeDutyCycle(0)
                            print("Go on, that is the right body")


                        # if angle < -20 and duty < 11:
                        #     duty = duty + 1
                        #     if angle < -40 and duty < 11:
                        #         duty = duty + 1
                        #         if angle < -60 and duty < 11:
                        #             duty = duty + 1
                        #     servo1.ChangeDutyCycle(duty)
                        #     time.sleep(0.5)
                        #     servo1.ChangeDutyCycle(0)

                        # if angle > 20 and duty > 3:
                        #     duty = duty - 1
                        #     if angle > 40 and duty > 3:
                        #         duty = duty - 1
                        #         if angle > 60 and duty > 3:
                        #             duty = duty - 1
                        #     servo1.ChangeDutyCycle(duty)
                        #     time.sleep(0.5)
                        #     servo1.ChangeDutyCycle(0)

                    else:
                        print("There is an AprilTag, but not 187")


                # Convert the dominant color to a web color name
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

                # Get the closest color name for the dominant color
                closest_color_name = rgb_to_name(dominant_color)

                # Print and display the dominant color information
                #print(f"Dominant Color: B:{dominant_color[0]}, G:{dominant_color[1]}, R:{dominant_color[2]}")
                #print(f"Closest Color: {closest_color_name}")
                
                message = f"{closest_color_name} shirt"
                
                # Calculate text size and position it in the middle of the trapezoid
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(message, font, 1, 2)[0]
                text_x = (roi.shape[1] - text_size[0]) // 2
                text_y = (roi.shape[0] + text_size[1]) // 2
                cv2.putText(frame, message, (top_left[0] + text_x, top_left[1] + text_y), font, 1, (0, 0, 255), 2)

                # Optionally, you can draw a rectangle around the specified region
                #cv2.rectangle(frame, top_left, bottom_right, average_color, thickness=3)
            else:
                # Handle the case when the ROI is empty or invalid
                print("Invalid or Empty ROI")
        #else:
            #print("One or more keypoints are missing.")
    # Draw 2D skeleton
    frame = renderer.draw(frame, body, draw_xyz)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        servo1.stop()
        GPIO.cleanup()
        print("Goodbye!")
        break

renderer.exit()
pose.exit()
