# Josh's IdentifAI Recruiting Challenge

## Development Setup

- **Python 3.9+**
- **FastAPI**
- Libraries for image processing and facial analysis: OpenCV, dlib, scipy, fastdtw

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/JoshFrancisCodes/recruiting-challenge.git
    ```

2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

3. Run the server locally:
    ```sh
    uvicorn app.main:app --reload
    ```

## Usage

Navigate to `http://127.0.0.1:8000/docs` to see the API documentation and interact with the API.

## API Endpoints

### Create Profile
```http
POST /create-profile
```
Upload an image to create a new profile. The profile will contain facial feature information and embeddings.
- **Response Model**: `Profile`

### Get Profile
```http
POST /get-profile
```
Upload an image to retrieve a profile. The profile will contain facial feature information and embeddings.
- **Response Model**: `Profile`

### Find Closest Profile
```http
POST /find-closest-profile
```
Upload an image and specify a facial feature (e.g., 'eyebrow', 'nose') to find the closest matching profile based on the shape similarity of that feature.
- **Query Parameter**: `feature` (Facial feature to compare)
- **Response Model**: `ProfileMatch`

### Measure Artifacts
```http
POST /measure-artifacts
```
Upload an image to measure various artifacts such as lighting inconsistency, blur, asymmetry, texture score, high-frequency artifacts, and gaze direction.
- **Response Model**: `Artifacts`

### Show Landmarks
```http
POST /show-landmarks
```
Upload an image to display the facial landmarks overlaid on the original image.
- **Response**: Image with landmarks

## Detailed Examples and Documentation

- **Profile Model Example**:
    ```json
    {
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
    ```

- **Artifacts Model Example**:
    ```json
    {
      "lighting_inconsistency": 0.1,
      "blur_measure": 0.2,
      "asymmetry_score": 0.3,
      "texture_score": 0.4,
      "high_freq_artifacts": 0.5,
      "gaze_direction": 0.6
    }
    ```

---