from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import tempfile
from face_encoder import FaceEncoder
from webcam import VideoFinder

app = FastAPI(title="Person Finder Backend")

# Load reference embedding once
encoder = FaceEncoder()
reference_embedding = encoder.encode("data/reference.jpg")

# For live webcam streaming
video_finder = VideoFinder(reference_embedding, video_source=0)

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file and run detection.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    finder = VideoFinder(reference_embedding, video_source=tmp_path)
    finder.run()  # Runs detection in window
    return {"status": "Detection completed"}

@app.get("/video_feed/")
def video_feed():
    """
    Stream live webcam with detection as MJPEG.
    """
    def generate():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = video_finder.detect_frame(frame)
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        cap.release()

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')
