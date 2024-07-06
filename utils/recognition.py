import cv2
import dlib
import numpy as np
from imutils import face_utils

# Initialize dlib's face detector, shape predictor, and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def get_face_embeddings(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale image
    rects = detector(gray, 1)
    embeddings = []
    for rect in rects:
        # Get the landmarks/parts for the face in box
        shape = predictor(gray, rect)
        #shape = face_utils.shape_to_np(shape)
        # Get the face descriptor
        face_descriptor = face_rec_model.compute_face_descriptor(image, shape)
        embeddings.append(np.array(face_descriptor))
    return embeddings

def compare_faces(embedding1, embedding2, threshold=0.6):
    # Compute the Euclidean distance between the two embeddings
    distance = np.linalg.norm(embedding1 - embedding2)
    return distance < threshold, distance

# # Load images
# image1 = cv2.imread('path_to_image1.jpg')
# image2 = cv2.imread('path_to_image2.jpg')

# # Get face embeddings for each image
# embeddings1 = get_face_embeddings(image1)
# embeddings2 = get_face_embeddings(image2)

# # Compare the first face found in each image
# if embeddings1 and embeddings2:
#     is_match, distance = compare_faces(embeddings1[0], embeddings2[0])
#     print(f"Match: {is_match}, Distance: {distance}")
# else:
#     print("Face not detected in one or both images")

# # Show images with detected faces
# for rect in detector(cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY), 1):
#     (x, y, w, h) = face_utils.rect_to_bb(rect)
#     cv2.rectangle(image1, (x, y), (x+w, y+h), (0, 255, 0), 2)

# for rect in detector(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), 1):
#     (x, y, w, h) = face_utils.rect_to_bb(rect)
#     cv2.rectangle(image2, (x, y), (x+w, y+h), (0, 255, 0), 2)

# cv2.imshow("Image 1", image1)
# cv2.imshow("Image 2", image2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
