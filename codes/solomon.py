import cv2
from math import atan2, degrees
import sys
import numpy as np
import apriltag
import RPi.GPIO as GPIO
import time
import argparse
import webcolors
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from mpl_toolkits.mplot3d import Axes3D
sys.path.append("..")
from BlazeposeDepthaiEdgeModified import BlazeposeDepthai
from BlazeposeRendererModified import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT
from collections import Counter

# Function to generate file name to avoid overwriting previous data
def generate_filename(base_name):
    counter = 0
    while True:
        file_name = f"{base_name}{'' if counter == 0 else str(counter)}.csv"
        if not os.path.exists(file_name):
            return file_name
        counter += 1
# Generate filename
csv_file_name = generate_filename("data")

# Function to calculate the average color of a region and outputs the rgb value
def calc_avg_color(region):
    pixels = region.reshape(-1, 3)
    average_color = np.uint8(pixels).mean(axis=0)
    average_color = (average_color[2], average_color[1], average_color[0])
    return average_color

# Function to Convert the average color to a web color name
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

# Function to start servo
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

# Create CSV file
with open(csv_file_name, 'w', newline='') as csvfile:
    fieldnames = ['time', 'x', 'y', 'z', 'angle_xz', 'tag', 'color', 'rgb', 'servo_angle']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='full',
                        help="Landmark model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
parser.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
args = parser.parse_args()  

# Initializing variables
start_time = 0
started_tracking = False
elapsed_time  = 0

distance_x, distance_y,distance_z = 0, 0, 0
angle = 0
apriltag_id = 'N/A'
closest_color_name = 'N/A'
average_color = (0, 0, 0)
servo_angle = 0
# array to store xyz values
xyz_values = []
occlusion = 0

# Initialize the servo
servo1, duty = initialize_servo()
pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model,xyz=args.xyz)
renderer = BlazeposeRenderer(pose, output=args.output)
detector = apriltag.Detector()

draw_xyz = False
while True:
    # Run blazepose on the nexray_framet frame
    frame, body, occlusion = pose.next_frame(occlusion, started_tracking)
    if frame is None:
        break

    # Get the coordinates of the shoulder and hip points
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

        # Define the four points of the quadrilateral
        points = [tuple(right_shoulder.astype(int)) if right_shoulder is not None else (0, 0),
                  tuple(left_shoulder.astype(int)) if left_shoulder is not None else (0, 0),
                  tuple(left_hip.astype(int)) if left_hip is not None else (0, 0),
                  tuple(right_hip.astype(int)) if right_hip is not None else (0, 0)]
        
        # the points
        # print(f"Debugging Landmarks: right_shoulder = {points[0]}, left_shoulder = {points[1]}, left_hip = {points[2]}, right_hip = {points[3]}")

        if all([right_shoulder is not None, left_shoulder is not None, 
                left_hip is not None, right_hip is not None]):
          
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

            # cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), thickness=3)
            # Crop the ROI within the rectangle
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # Check if the ROI is empty or invalid
            if roi.size > 0:
                gray_cropped_img = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                h, w, _ = roi.shape
                mid_h, mid_w = h // 2, w // 2

                # Divide the ROI into 4 parts
                roi1 = roi[:mid_h, :mid_w]
                roi2 = roi[:mid_h, mid_w:]
                roi3 = roi[mid_h:, :mid_w]
                roi4 = roi[mid_h:, mid_w:]
                average_color = calc_avg_color(roi)
                avg_color1 = calc_avg_color(roi1)
                avg_color2 = calc_avg_color(roi2)
                avg_color3 = calc_avg_color(roi3)
                avg_color4 = calc_avg_color(roi4)

                # If there is occlusion or not started tracking, check for tags
                if occlusion >  0 or not started_tracking:
                    
                    if not started_tracking:
                        print("Please show your tag for 5 seconds to begin")
                    if started_tracking:
                        print("There is occlusion please show tag")
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
                    # get the xyz
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

                    # Define angle thresholds and the duty increment/decrement value
                    angle_thresholds = [-45, -30, -15, 15, 30, 45]
                    duty_delta = 0.75

                    # Initialize old_duty to track previous duty value
                    old_duty = duty

                    # Handle negative angles
                    if angle <= -15 and duty < 11:
                        for threshold in reversed(angle_thresholds[:3]):
                            if angle < threshold:
                                duty += duty_delta

                    # Handle positive angles
                    elif angle >= 15 and duty > 3:
                        for threshold in angle_thresholds[3:]:
                            if angle > threshold:
                                duty -= duty_delta
  
                    # Update the servo
                    if old_duty != duty:
                        servo1.ChangeDutyCycle(duty)
                        time.sleep(0.5)
                        servo1.ChangeDutyCycle(0)
                        servo_angle = (duty - 2.5) * 20
                        print(f"Servo moved rotates to angle {servo_angle}")

                # Get the closest color name for the average color
                closest_color_name = rgb_to_name(average_color)
                message = f"{closest_color_name} shirt"
                
                # Calculate text size and position it in the middle of the trapezoid
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(message, font, 1, 2)[0]
                text_x = (roi.shape[1] - text_size[0]) // 2
                text_y = (roi.shape[0] + text_size[1]) // 2
                cv2.putText(frame, message, (top_left[0] + text_x, top_left[1] + text_y), font, 1, (0, 0, 255), 2)
            else:
                # Handle the case when the ROI is empty or invalid
                print("Invalid or Empty ROI")
        #else:
            #print("One or more keypoints are missing.")
        # Store the xyz values
        xyz_values.append([distance_x, distance_y, distance_z])
         # collect data:     
        with open(csv_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'time': elapsed_time, 'x': distance_x, 'y': distance_y, 'z': distance_z, 
                            'angle_xz': angle, 'tag': apriltag_id, 'color': closest_color_name, 'rgb': average_color, 
                            'servo_angle': servo_angle})

    # Draw 2D skeleton
    frame = renderer.draw(frame, body, draw_xyz)
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        servo1.ChangeDutyCycle(7)
        time.sleep(0.5)
        servo1.stop()
        GPIO.cleanup()

        # for 3d graph
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xyz_values = np.array(xyz_values)
        # ax.scatter(xyz_values[:, 0], xyz_values[:, 1], xyz_values[:, 2])
        # plt.show()

        # # Create a new figure
        # fig = plt.figure()
        # Line plot
        ax.plot(xyz_values[:, 0], xyz_values[:, 1], xyz_values[:, 2])
        plt.show()
        # # Load your data
        # df = pd.read_csv(csv_file_name)

        # # Plotting body landmark x, y over time
        # plt.figure()
        # plt.plot(df['time'], df['x'], label='X Coordinate')
        # plt.plot(df['time'], df['z'], label='z Coordinate')
        # plt.legend()
        # plt.title('Body Landmark Coordinates over Time')
        # plt.show()

        # # Histogram of angles
        # plt.figure()
        # plt.hist(df['angle_xz'], bins=50)
        # plt.title('Histogram of Angles')
        # plt.show()

        print("Goodbye!")
        break

renderer.exit()
pose.exit()
