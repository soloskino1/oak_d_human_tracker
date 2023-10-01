import cv2
from math import atan2, degrees
import sys
import numpy as np
sys.path.append("../..")
from BlazeposeDepthai import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT
import argparse
from sklearn.cluster import KMeans
import webcolors
from collections import Counter
import colorspacious as cs
from scipy.stats import mode
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'


# For gesture demo
semaphore_flag = {
        (3,4):'A', (2,4):'B', (1,4):'C', (0,4):'D',
        (4,7):'E', (4,6):'F', (4,5):'G', (2,3):'H',
        (0,3):'I', (0,6):'J', (3,0):'K', (3,7):'L',
        (3,6):'M', (3,5):'N', (2,1):'O', (2,0):'P',
        (2,7):'Q', (2,6):'R', (2,5):'S', (1,0):'T',
        (1,7):'U', (0,5):'V', (7,6):'W', (7,5):'X',
        (1,6):'Y', (5,6):'Z',
}


def perform_ocr_in_roi(roi):
    # Convert the ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Perform OCR on the grayscale ROI
    text = pytesseract.image_to_string(gray_roi)

    return text

def recognize_gesture(b):  
    # b: body         

    def angle_with_y(v):
        # v: 2d vector (x,y)
        # Returns angle in degree of v with y-axis of image plane
        if v[1] == 0:
            return 90
        angle = atan2(v[0], v[1])
        return degrees(angle)

    # For the demo, we want to recognize the flag semaphore alphabet
    # For this task, we just need to measure the angles of both arms with vertical
    right_arm_angle = angle_with_y(b.landmarks[KEYPOINT_DICT['right_elbow'],:2] - b.landmarks[KEYPOINT_DICT['right_shoulder'],:2])
    left_arm_angle = angle_with_y(b.landmarks[KEYPOINT_DICT['left_elbow'],:2] - b.landmarks[KEYPOINT_DICT['left_shoulder'],:2])
    right_pose = int((right_arm_angle +202.5) / 45) % 8 
    left_pose = int((left_arm_angle +202.5) / 45) % 8
    letter = semaphore_flag.get((right_pose, left_pose), None)
    return letter

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='full',
                        help="Landmark model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  
parser.add_argument("-o","--output",
                    help="Path to output video file")
args = parser.parse_args()            

pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model)
renderer = BlazeposeRenderer(pose, output=args.output)

while True:
    # Run blazepose on the next frame
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

        # Draw the quadrilateral (green color)
        #cv2.polylines(frame, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Determine the top-left corner of the rectangle (right shoulder)
        top_left = tuple(right_shoulder.astype(int))

        # Determine the bottom-right corner of the rectangle
        bottom_right = (left_shoulder[0], left_hip[1])

        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), thickness=3)
        # Crop the ROI within the rectangle
        roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        # Check if the ROI is empty or invalid
        if roi.size > 0:
            # Reshape the ROI into an array of pixel values
            pixels = roi.reshape(-1, 3)
            # Calculate the mode (most frequent color) in the ROI
            dominant_color = np.uint8(pixels).mean(axis=0)

            # Convert the dominant color to a tuple (B, G, R)
            dominant_color = (dominant_color[2], dominant_color[1], dominant_color[0])



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
            print(f"Dominant Color: B:{dominant_color[0]}, G:{dominant_color[1]}, R:{dominant_color[2]}")
            print(f"Closest Color: {closest_color_name}")
            
            message = f"{closest_color_name} shirt"
            # Calculate text size and position it in the middle of the trapezoid
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(message, font, 1, 2)[0]
            text_x = (roi.shape[1] - text_size[0]) // 2
            text_y = (roi.shape[0] + text_size[1]) // 2
            cv2.putText(frame, message, (top_left[0] + text_x, top_left[1] + text_y), font, 1, (0, 0, 255), 2)
            
            # Perform OCR in the ROI
            ocr_text = perform_ocr_in_roi(roi)

            # Print and display the recognized text
            print(f"Recognized Text: {ocr_text}")

            # Draw the recognized text on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, ocr_text, (top_left[0], top_left[1] - 10), font, 0.5, (0, 0, 255), 2)

            # Optionally, you can draw a rectangle around the specified region
            #cv2.rectangle(frame, top_left, bottom_right, average_color, thickness=3)
        else:
            # Handle the case when the ROI is empty or invalid
            print("Invalid or Empty ROI")

    # Gesture recognition
    if body:
        letter = recognize_gesture(body)
        if letter:
            cv2.putText(frame, letter, (frame.shape[1] // 2, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 190, 255), 3)

    # Draw 2D skeleton
    frame = renderer.draw(frame, body)
    
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break

renderer.exit()
pose.exit()

