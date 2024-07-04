import numpy as np

def measure_asymmetry(image, landmarks):
    left_side = landmarks[:len(landmarks)//2]
    right_side = landmarks[len(landmarks)//2:]
    right_side = np.flip(right_side, axis=0)

    distances = np.linalg.norm(left_side - right_side, axis=1)
    asymmetry_score = np.mean(distances)
    return asymmetry_score
