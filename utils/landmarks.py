import cv2
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        return shape
    return None

def draw_landmarks(image, landmarks):
    height, width = image.shape[:2]
    radius = max(1, int(min(width, height) * 0.005))
    font_scale = radius * 0.3  # Font scale proportional to radius
    thickness = max(1, int(radius * 0.5))  # Thickness proportional to radius

    for i, (x, y) in enumerate(landmarks):
        cv2.circle(image, (x, y), radius, (0, 255, 0), -1)
        cv2.putText(image, str(i), (x + radius, y - radius), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
    
    return image

def extract_shapes(landmarks):
    shapes = {
        'left_eyebrow': landmarks[17:22],
        'right_eyebrow': landmarks[22:27],
        'left_eye': landmarks[36:42],
        'right_eye': landmarks[42:48],
        'nose': landmarks[27:36],
        'mouth': landmarks[48:],
        'jaw': landmarks[0:17]
    }
    return shapes