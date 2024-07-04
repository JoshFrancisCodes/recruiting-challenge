def determine_face_shape(landmarks):
    jaw_width = landmarks[16][0] - landmarks[0][0]
    face_height = landmarks[8][1] - landmarks[19][1]
    cheekbone_width = landmarks[15][0] - landmarks[1][0]
    print(jaw_width, face_height, cheekbone_width)
    if face_height / jaw_width > 1.5:
        return "Oval"
    elif cheekbone_width > jaw_width:
        return "Round"
    else:
        return "Square"

def analyze_cheekbones(landmarks):
    cheekbone_prominence = landmarks[14][1] - landmarks[3][1]
    if cheekbone_prominence > 10:
        return "High cheekbones"
    else:
        return "Average cheekbones"
