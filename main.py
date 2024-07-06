from fastapi import FastAPI, File, UploadFile, HTTPException, Query
import uvicorn
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO
from typing import List

from utils import get_landmarks, get_image, draw_landmarks, get_face_embeddings, compare_faces, extract_shapes, procrustes_analysis, hausdorff_distance, dtw_distance
from features import create_profile_description
from detectors import measure_lighting_inconsistency, measure_blur, measure_asymmetry, measure_skin_texture, measure_high_frequency_artifacts, measure_gaze_inconsistency

app = FastAPI(
    title="Josh's Facial Profile API Documentation",
    description="This API contains various endpoints for facial profile management and matching.\n \
        It includes operations for creating, retrieving, and managing profiles, as well as operations for matching profiles based on facial features.\n \
            Finally, it includes data on artifacts in images, which can be useful for deepfake detection.",
    version="1.0.0",
    contact={
        "name": "Josh Francis",
        "url": "https://www.linkedin.com/in/josh--francis/",
        "email": "josfran@stanford.edu",
    },
    # license_info={
    #     "name": "MIT License",
    #     "url": "https://opensource.org/licenses/MIT",
    # },
    openapi_tags=[
        {
            "name": "Profile Matching",
            "description": "Operations related to matching profiles based on facial features.",
        },
        {
            "name": "Profile Management",
            "description": "Operations related to creating, retrieving, and managing profiles.",
        },
    ],)

class Artifacts(BaseModel):
    """
    Stores data about how many artifacts are present in an image.
    These artifacts are often highly present in deepfaked images,
    so loooking at their prevalence in an image relative to the average
    amounts in a dataset can be useful in deepfake detection.
    """
    lighting_inconsistency: float
    blur_measure: float
    asymmetry_score: float
    texture_score: float
    high_freq_artifacts: float
    gaze_direction: float
    
class Profile(BaseModel):
    """
    Represents a profile of a face.
    Has an integer identifier, text description, face embedding, artifacts, and landmarks.
    Landmarks are produced by dlib shape predictor model.
    Text description is calculated based on landmarks.
    Artifacts are calculated using OpenCV methods.
    """
    id: int = -1
    description: str
    embedding: List[float]
    artifacts: Artifacts
    landmarks: List[List[int]]
    
    class Config:
        schema_extra = {
            "example": {
                "id": 1,
                "description": "Face with high cheekbones, square shape, and blue eyes.",
                "embedding": [0.1, 0.2, 0.3],
                "artifacts": {
                    "lighting_inconsistency": 0.1,
                    "blur_measure": 0.2,
                    "asymmetry_score": 0.3,
                    "texture_score": 0.4,
                    "high_freq_artifacts": 0.5,
                    "gaze_direction": 0.6
                },
                "landmarks": [[0, 0], [1, 1], [2, 2]]
            }
        }
    
class ProfileMatch(BaseModel):
    current_profile: Profile
    found_profile: Profile
    distance: float
    
class ProfileDistance(BaseModel):
    artifacts_distance: Artifacts
    embedding_distance: float
    eyebrow_distance: float
    eye_distance: float
    nose_distance: float
    mouth_distance: float
    jaw_distance: float

profiles: List[Profile] = []
    
@app.get("/list-profiles", response_model=List[Profile], tags=["Profile Management"])
async def list_profiles():
    """
    _List all profiles._

    Returns:
        _[Profile]_: _List of all profiles._
    """
    return profiles

@app.get("/count-profiles", tags=["Profile Management"])
async def count_profiles():
    """
    _Count the number of profiles._

    Returns:
        _int_: _The number of profiles._
    """
    return len(profiles)

@app.get("/get-profile/{profile_id}", response_model=Profile, tags=["Profile Management"])
async def get_profile_by_id(profile_id: int):
    """
    _Get a profile by id._

    Args:
        profile_id (int): _The id of the profile to get._

    Raises:
        HTTPException: _404 if the profile is not found_

    Returns:
        _int_: _description_
    """
    for profile in profiles:
        if profile.id == profile_id:
            return profile
    raise HTTPException(status_code=404, detail="Profile not found")

@app.delete("/delete-profile/{profile_id}", tags=["Profile Management"])
async def delete_profile(profile_id: int):
    """
    _Delete a profile by id._

    Args:
        profile_id (int): _The id of the profile to delete._

    Raises:
        HTTPException: _404 if the profile is not found_
    """
    for i, profile in enumerate(profiles):
        if profile.id == profile_id:
            del profiles[i]
            return
    raise HTTPException(status_code=404, detail="Profile not found")

@app.delete("/delete-all-profiles", tags=["Profile Management"])
async def delete_all_profiles():
    """
    _Delete all profiles._
    """
    profiles.clear()

@app.post("/get-profile-id", response_model=int, tags=["Profile Management"])
async def get_profile_id(file: UploadFile = File(...)):
    """
    _Get the id of the profile of the face in the image you upload._

    Args:
        file (UploadFile, optional): _The image to get the id of._ Defaults to File(...).

    Raises:
        HTTPException: _404 if no matching profile is found_
        HTTPException: _400 if no face is detected_

    Returns:
        _int_: _The id of the profile of the face in the image._
    """
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        current_profile = create_profile_helper(img, landmarks)
        for profile in profiles:
            if np.array_equal(profile.embedding, current_profile.embedding):
                return profile.id
        raise HTTPException(status_code=404, detail="Profile not found")
    else:
        raise HTTPException(status_code=400, detail="No face detected")

@app.post("/create-profile", response_model=Profile, tags=["Profile Management"])
async def create_profile(file: UploadFile = File(...)):
    """
    _Create a profile based on the face in the image you upload_

    Args:
        file (UploadFile, optional): _The image to get the profile from._ Defaults to File(...).
        
    Raises:
        HTTPException: _400 if no face is detected._

    Returns:
        _Profile_: _The profile of the face in the image._
    """
    return await get_profile(True, file)
    
@app.post("/get-profile", response_model=Profile, tags=["Profile Management"])
async def get_profile(file: UploadFile = File(...)):
    """
    _Get the profile of the face in the image you upload._

    Args:
        file (UploadFile, optional): _The image to get the profile from._ Defaults to File(...).
        
    Raises:
        HTTPException: _400 if no matching profile is found_

    Returns:
        _Profile_: _The profile of the face in the image._
    """
    return await get_profile(False, file)
    
async def get_profile(create: bool, file: UploadFile = File(...), tags=["Profile Management"]):
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        profile = create_profile_helper(img, landmarks)
        if create: 
            profile.id = len(profiles)
            profiles.append(profile)
        return profile
    else:
        raise HTTPException(status_code=400, detail="No face detected")
    
@app.post("/find-profile", response_model=ProfileMatch, tags=["Profile Matching"])
async def find_profile(file: UploadFile = File(...)):
    """
    _Find the closest matching profile based on face embeddings._

    Args:
        file (UploadFile, optional): _The image to find the closest matching profile of._ Defaults to File(...).

    Raises:
        HTTPException: _401 if no matching profile is found_
        HTTPException: _400 if no face is detected_

    Returns:
        _ProfileMatch_: _The profile of the image you uploaded, the closest matching profile, and the distance between the two profiles._
    """
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        current_profile = create_profile_helper(img, landmarks)
        embedding = np.array(current_profile.embedding)
        
        min_distance = float("inf")
        closest_profile = None
        for profile in profiles:
            stored_embedding = np.array(profile.embedding)
            _, distance = compare_faces(embedding, stored_embedding)
            if distance < min_distance:
                min_distance = distance
                closest_profile = profile
                
        if closest_profile is not None:
            return ProfileMatch(current_profile=current_profile, found_profile=closest_profile, distance=min_distance)
        else:
            raise HTTPException(status_code=401, detail="No matching profile found")
    else:
        raise HTTPException(status_code=400, detail="No face detected")
    
@app.post("/compare-profiles", response_model=ProfileDistance, tags=["Profile Matching"])
async def compare_profiles(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    """
    _Upload two images to compare the profiles of the faces in the images.
    Profiles are compared based on facial features, artifacts, and face embeddings._

    Args:
        file1 (UploadFile, optional): _First image to compare._ Defaults to File(...).
        file2 (UploadFile, optional): _Second image to compare._ Defaults to File(...).

    Raises:
        HTTPException: _Raises an error if no face is detected in either image._

    Returns:
        _ProfileDistance_: _Distance between the two profiles._
    """
    img1 = await get_image(file1)
    img2 = await get_image(file2)
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)
    if landmarks1 is not None and landmarks2 is not None:
        return calculate_distance(img1, img2, landmarks1, landmarks2)
    else:
        raise HTTPException(status_code=400, detail="No face detected")

def calculate_distance(img1, img2, landmarks1, landmarks2): 
    # embedding distance
    embedding1 = np.array(get_face_embeddings(img1)[0])
    embedding2 = np.array(get_face_embeddings(img2)[0])
    embedding_distance = compare_faces(embedding1, embedding2)[1]
    
    # artifacts distance
    lighting_inconsistency = abs(measure_lighting_inconsistency(img1) - measure_lighting_inconsistency(img2))
    blur_measure = abs(measure_blur(img1) - measure_blur(img2))
    asymmetry_score = abs(measure_asymmetry(img1, landmarks1) - measure_asymmetry(img2, landmarks2))
    texture_score = abs(measure_skin_texture(img1) - measure_skin_texture(img2))
    high_freq_artifacts = abs(measure_high_frequency_artifacts(img1) - measure_high_frequency_artifacts(img2))
    gaze_direction = abs(measure_gaze_inconsistency(landmarks1) - measure_gaze_inconsistency(landmarks2))
    artifacts_distance = Artifacts(lighting_inconsistency=lighting_inconsistency, blur_measure=blur_measure, asymmetry_score=asymmetry_score, 
                                   texture_score=texture_score, high_freq_artifacts=high_freq_artifacts, gaze_direction=gaze_direction)
    
    # shape distance
    shapes1 = extract_shapes(landmarks1)
    shapes2 = extract_shapes(landmarks2)
    eyebrow_distance = dtw_distance(shapes1['left_eyebrow'], shapes2['left_eyebrow']) + dtw_distance(shapes1['right_eyebrow'], shapes2['right_eyebrow'])
    eye_distance = procrustes_analysis(shapes1['left_eye'], shapes2['left_eye']) + procrustes_analysis(shapes1['right_eye'], shapes2['right_eye'])
    nose_distance = procrustes_analysis(shapes1['nose'], shapes2['nose'])
    mouth_distance = procrustes_analysis(shapes1['mouth'], shapes2['mouth'])
    jaw_distance = dtw_distance(shapes1['jaw'], shapes2['jaw'])
    
    distance = ProfileDistance(artifacts_distance=artifacts_distance, embedding_distance=embedding_distance, 
                               eyebrow_distance=eyebrow_distance, eye_distance=eye_distance, nose_distance=nose_distance, mouth_distance=mouth_distance, jaw_distance=jaw_distance)
    
    return distance 

@app.post("/find-closest-profile", response_model=ProfileMatch, tags=["Profile Matching"])
async def find_closest_profile(file: UploadFile = File(...), feature: str = Query(..., description="Facial feature to compare ('right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye', 'nose', 'mouth', 'jaw')")):
    """
    _Find the closest matching profile based on a specific facial feature._

    Args:
        file (UploadFile, optional): _Image to find the closest profile to._ Defaults to File(...).
        feature (str, optional): _Name of the feature to query on_. Defaults to Query(..., description="Facial feature to compare ('right_eyebrow', 'left_eyebrow', 'right_eye', 'left_eye', 'nose', 'mouth', 'jaw')").

    Raises:
        HTTPException: _404 if no matching profile is found_
        HTTPException: _400 if no face is detected_
        HTTPException: _401 if an invalid feature is specified_

    Returns:
        _ProfileMatch_: _The profile of the image you uploaded, the closest matching profile, and the distance between the two profiles._
    """
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        shapes = extract_shapes(landmarks)
        if feature not in shapes:
            raise HTTPException(status_code=401, detail="Invalid feature specified")
        
        current_shape = shapes[feature]
        min_distance = float("inf")
        closest_profile = None
        for profile in profiles:
            stored_shape = np.array(extract_shapes(profile.landmarks)[feature])
            distance = dtw_distance(current_shape, stored_shape) if feature in ['left_eyebrow', 'right_eyebrow', 'jaw'] else procrustes_analysis(current_shape, stored_shape)
            if distance < min_distance:
                min_distance = distance
                closest_profile = profile

        if closest_profile is not None:
            current_profile = create_profile_helper(img, landmarks)
            return ProfileMatch(current_profile=current_profile, found_profile=closest_profile, distance=min_distance)
        else:
            raise HTTPException(status_code=404, detail="No matching profile found")
    else:
        raise HTTPException(status_code=400, detail="No face detected")
    

@app.post("/measure-artifacts", response_model=Artifacts, tags=["Profile Matching"])
async def measure_artifacts(file: UploadFile = File(...)):
    """
    _Measure artifacts in the image you upload.
    Artifacts are chosen based on their prevalence in
    deepfaked photos._

    Args:
        file (UploadFile, optional): The image to analyze the artifacts in. Defaults to File(...).

    Raises:
        HTTPException: _400 if no face is detected_

    Returns:
        _Artifacts_: _The artifacts in the image._
    """
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        artifact_scores = get_artifacts(img, landmarks)
        return artifact_scores
    else:
        raise HTTPException(status_code=400, detail="No face detected")
    
@app.post("/show-landmarks", tags=["Profile Matching"])
async def show_landmarks(file: UploadFile = File(...)):
    """
    _Show the landmarks of the face in the image you upload._

    Args:
        file (UploadFile, optional): _Image to show the facial landmarks on top of_. Defaults to File(...).

    Raises:
        HTTPException: _400 if no face is detected_

    Returns:
        _StreamingResponse_: _Image with facial landmarks drawn on top._
    """
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        img_with_landmarks = draw_landmarks(img, landmarks)
        _, img_encoded = cv2.imencode('.png', img_with_landmarks)
        return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/png")
    else:
        raise HTTPException(status_code=400, detail="No face detected")

def get_artifacts(img, landmarks):
    return Artifacts(
            lighting_inconsistency=measure_lighting_inconsistency(img),
            blur_measure=measure_blur(img),
            asymmetry_score=measure_asymmetry(img, landmarks),
            texture_score=measure_skin_texture(img),
            high_freq_artifacts=measure_high_frequency_artifacts(img),
            gaze_direction=measure_gaze_inconsistency(landmarks)
        
        )
    
def create_profile_helper(img, landmarks):
    profile_description = create_profile_description(img, landmarks)
    embedding = get_face_embeddings(img)[0]
    artifacts = get_artifacts(img, landmarks)
    # print the type of landmarks for debugging purposes
    return Profile(description=profile_description, embedding=embedding.tolist(), artifacts=artifacts, landmarks=landmarks.tolist())

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
