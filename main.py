from fastapi import FastAPI, File, UploadFile
import uvicorn
from pydantic import BaseModel
from starlette.responses import StreamingResponse
import numpy as np
import cv2
from io import BytesIO

from utils import get_landmarks, get_image, draw_landmarks
from features import create_profile_description

app = FastAPI()

class Profile(BaseModel):
    description: str

@app.post("/show-landmarks")
async def show_landmarks(file: UploadFile = File(...)):
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        img_with_landmarks = draw_landmarks(img, landmarks)
        _, img_encoded = cv2.imencode('.png', img_with_landmarks)
        return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/png")
    else:
        return {"description": "No face detected"}

@app.post("/create-profile", response_model=Profile)
async def create_profile(file: UploadFile = File(...)):
    img = await get_image(file)
    landmarks = get_landmarks(img)
    if landmarks is not None:
        profile_description = create_profile_description(img, landmarks)
        return {"description": profile_description}
    else:
        return {"error": "No face detected"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
