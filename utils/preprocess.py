from io import BytesIO
from PIL import Image
import numpy as np
import cv2

async def get_image(file): 
    img = Image.open(BytesIO(await file.read()))
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img