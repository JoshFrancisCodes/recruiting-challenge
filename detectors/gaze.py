import numpy as np

def measure_gaze_inconsistency(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    eye_vector = right_eye_center - left_eye_center
    gaze_direction = np.arctan2(eye_vector[1], eye_vector[0])
    return gaze_direction
