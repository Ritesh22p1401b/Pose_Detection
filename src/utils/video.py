import cv2

def open_video(source):
    return cv2.VideoCapture(source)

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame
