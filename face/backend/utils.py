import cv2
import os
from datetime import datetime

def save_frame(frame, folder="output_frames", prefix="frame"):
    """
    Save a single frame as an image file.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(folder, f"{prefix}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename

def save_video(frames, output_path="output.avi", fps=20.0, frame_size=None):
    """
    Save a list of frames as a video.
    frames: list of np.ndarray
    frame_size: (width, height), if None inferred from first frame
    """
    if not frames:
        raise ValueError("No frames to save")

    if frame_size is None:
        height, width = frames[0].shape[:2]
        frame_size = (width, height)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        out.write(frame)

    out.release()
    return output_path

def log_detection(person_name="Unknown", folder="logs"):
    """
    Log a detection event with timestamp.
    """
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(folder, "detections.txt")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} - {person_name} detected\n")
    return log_file
