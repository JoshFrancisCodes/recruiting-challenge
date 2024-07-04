import cv2
import numpy as np

def get_eye_color(image, landmarks):
    left_eye_region = image[landmarks[37][1]:landmarks[41][1], landmarks[36][0]:landmarks[39][0]]
    right_eye_region = image[landmarks[43][1]:landmarks[47][1], landmarks[42][0]:landmarks[45][0]]

    left_eye_color = get_dominant_color(left_eye_region)
    right_eye_color = get_dominant_color(right_eye_region)

    if left_eye_color == right_eye_color:
        return left_eye_color
    else:
        return "Mixed"

def get_dominant_color(image_region):
    # Convert to HSV and calculate histogram
    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    dominant_color = cv2.minMaxLoc(hist)[3][1]
    print(cv2.minMaxLoc(hist))
    return interpret_color(dominant_color)

def interpret_color(dominant_color):
    # normally 0 - 60 is red, 60 - 180 is green, 180 - 240 is blue, and 240-360 is red again
    # but we're in 0 - 180 so halve everything
    if dominant_color < 20 or dominant_color > 160:
        return "Brown"
    elif dominant_color < 30 or dominant_color > 150:
        return "Light Brown"
    elif dominant_color < 90:
        return "Green"
    else:
        return "Blue"
