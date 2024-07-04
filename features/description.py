from .shape import determine_face_shape, analyze_cheekbones
from .color import get_eye_color

def create_profile_description(image, landmarks):
    face_shape = determine_face_shape(landmarks)
    cheekbones = analyze_cheekbones(landmarks)
    eye_color = get_eye_color(image, landmarks)

    profile_description = f"Face with {cheekbones}, {face_shape} shape, and {eye_color} eyes."
    return profile_description
