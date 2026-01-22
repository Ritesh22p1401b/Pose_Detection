import cv2
from camera.camera_manager import CameraManager

def main():
    print("[INFO] Starting phone camera test...")

    # Change to "laptop" if needed
    cam = CameraManager(source="phone")

    try:
        cam.open()
    except RuntimeError as e:
        print("[ERROR]", e)
        return

    print("[INFO] Camera opened. Press Q or ESC to exit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        cv2.imshow("Camera Test (Phone)", frame)

        key = cv2.waitKey(1)
        if key in (ord('q'), 27):
            break

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera test finished.")

if __name__ == "__main__":
    main()
