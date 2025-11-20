from ultralytics import YOLO
import cv2
import os

# Load model
model = YOLO("models/best.pt")

# Force a custom output folder (always created)
OUTPUT_DIR = "results_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def detect_image(img_path):
    print(f"\n Detecting image: {img_path}")

    # Run detection + force save
    results = model(img_path, conf=0.4, save=True, project=OUTPUT_DIR, name="images")

    from pathlib import Path
    save_dir = Path(results[0].save_dir)

    # Final saved file path
    saved_img_path = save_dir / os.path.basename(img_path)

    print(f"➡ Saved to: {saved_img_path}")

    # Load and display
    img = cv2.imread(str(saved_img_path))
    if img is None:
        print(" ERROR: Saved image NOT found or cannot be loaded.")
        return

    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(" Image detection complete.")



#  VIDEO DETECTION
def detect_video(video_path):
    print(f"\n Detecting video: {video_path}")

    results = model(video_path, conf=0.4, save=True, project=OUTPUT_DIR, name="videos")

    print(f"➡ Output saved to: {results[0].save_dir}")
    print(" The processed video will appear in that folder.")
    print(" Video detection complete.")


#  WEBCAM DETECTION
def detect_webcam():
    print("\n Starting webcam...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(" ERROR: Webcam not detected.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)

        annotated = results[0].plot()

        cv2.imshow("Webcam PPE Detection", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Webcam detection stopped.")



#  MAIN
if __name__ == "__main__":
    print("\n=== PPE DETECTOR STARTED ===")

    detect_image("sample_detect4.jpg")
    #detect_video("hardhat.mp4")
    #detect_webcam()

    print(" Edit the last 3 lines in detect.py to choose image/video/webcam.\n")