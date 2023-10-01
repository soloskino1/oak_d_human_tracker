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
            # Calculate the average color within the ROI
            average_color = np.mean(roi, axis=(0, 1))

            # Convert the average color to integers (RGB format)
            average_color = tuple(map(int, average_color))

            # Now, average_color contains the main color of the specified region in RGB format
            print(f"Average Color: R:{average_color[2]}, G:{average_color[1]}, B:{average_color[0]}")

            # Optionally, you can draw a rectangle around the specified region
            cv2.rectangle(frame, top_left, bottom_right, average_color, thickness=3)
        else:
            # Handle the case when the ROI is empty or invalid
            print("Invalid or Empty ROI")


        """
        # Extract the region within the quadrilateral
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [np.array(points)], (255, 255, 255))  # White mask in the quadrilateral shape
        region = cv2.bitwise_and(frame, mask)
        """


        """
        # Convert the region to HSV color space (Hue, Saturation, Value)
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        # Calculate the color histogram of the region
        histogram = cv2.calcHist([hsv_region], [0], None, [256], [0, 256])

        # Set a minimum saturation threshold (adjust as needed)
        min_saturation = 50

        # Find the most dominant color by finding the bin with the highest frequency
        dominant_color_bin = np.argmax(histogram)

        # Map the bin value to a specific color
        def bin_to_color(bin_value):
            hue = int((bin_value / 256) * 180)  # Map to the hue range (0-180)
            return hue

        dominant_hue = bin_to_color(dominant_color_bin)

        # Determine the color name based on the dominant hue and saturation
        def hue_to_color_name(hue_value, saturation_value):
            # Define hue and saturation ranges for common colors
            color_ranges = {
                (0, 30): ("Red", 100),
                (30, 90): ("Yellow", 100),
                (90, 150): ("Green", 100),
                (150, 210): ("Cyan", 100),
                (210, 270): ("Blue", 100),
                (270, 330): ("Magenta", 100),
                (330, 360): ("Red", 100),
            }

            for hue_range, (color_name, min_sat) in color_ranges.items():
                if hue_value >= hue_range[0] and hue_value < hue_range[1] and saturation_value >= min_sat:
                    return color_name

            return "Unknown"

        # Get the saturation value of the dominant color
        dominant_saturation = np.max(hsv_region[:, :, 1])

        # Check if the dominant color meets the minimum saturation threshold
        if dominant_saturation >= min_saturation:
            web_color_name = hue_to_color_name(dominant_hue, dominant_saturation)
        else:
            web_color_name = "Unknown"
            
            
        # Now, integrate the provided code for displaying dominant colors
        n_clusters = 5

        # To reduce complexity, resize the image
        data = cv2.resize(frame, (100, 100)).reshape(-1, 3)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(data.astype(np.float32), n_clusters, None, criteria, 10, flags)

        cluster_sizes = np.bincount(labels.flatten())

        palette = []
        for cluster_idx in np.argsort(-cluster_sizes):
            palette.append(np.full((frame.shape[0], frame.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))
        palette = np.hstack(palette)

        sf = frame.shape[1] / palette.shape[1]
        out = np.vstack([frame, cv2.resize(palette, (0, 0), fx=sf, fy=sf)])

        cv2.imshow("dominant_colors", out)
        cv2.waitKey(1)

        message = f"This person is wearing a {web_color_name} shirt"
        """

        """ pixels = region.reshape((-1, 3))

        # Use K-Means clustering to find dominant colors
        num_clusters = 1  # You can adjust the number of clusters as needed
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_.astype(np.uint8)



        web_color_name = rgb_to_name(dominant_color[0])

        message = f"This person is wearing a {web_color_name} shirt"
        
        """
        
        """
        # Calculate text size and position it in the middle of the trapezoid
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(message, font, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2
        cv2.putText(frame, message, (text_x, text_y), font, 1, (0, 0, 255), 2)
        """
        """
                # Calculate the majority color in the region (you can define your black range)
                black_lower = np.array([0, 0, 0], dtype=np.uint8)
                black_upper = np.array([50, 50, 50], dtype=np.uint8)
                black_mask = cv2.inRange(region, black_lower, black_upper)
                black_percentage = np.count_nonzero(black_mask) / (region.shape[0] * region.shape[1])

                # Render the frame only if the majority color is black
                if black_percentage > 0.1:  # You can adjust the threshold as needed
                    message = "This person is not wearing a black shirt"
                    if black_percentage < 0.1:
                        message = "This person is wearing a black shirt"
                    # Calculate text size and position it in the middle of the trapezoid
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text_size = cv2.getTextSize(message, font, 1, 2)[0]
                    text_x = (frame.shape[1] - text_size[0]) // 2
                    text_y = (frame.shape[0] + text_size[1]) // 2
                    cv2.putText(frame, message, (text_x, text_y), font, 1, (0, 0, 255), 2)


        """

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
