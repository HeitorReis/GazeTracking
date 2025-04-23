import cv2
from gaze_tracking import GazeTracking
import time
import numpy as np

def get_center_distance(face, frame_center):
    x, y, w, h = face
    face_center = (x + w // 2, y + h // 2)
    return ((face_center[0] - frame_center[0]) ** 2 + (face_center[1] - frame_center[1]) ** 2) ** 0.5


def enhance_image(img):
    # Work on a copy to avoid in-place changes
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_RGB = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    yuv = cv2.cvtColor(gray_RGB, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(blurred, -1, kernel)
    
    filtered = cv2.bilateralFilter(sharpened, 9, 75, 75)
    return filtered


def main():
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Camera error.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        cv2.namedWindow("Gaze Tracker", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Gaze Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # If faces found, select the one closest to center
        if len(faces) > 0:
            h, w = frame.shape[:2]
            frame_center = (w // 2, h // 2)
            
            closest_face = min(faces, key=lambda face: get_center_distance(face, frame_center))
            x, y, fw, fh = closest_face
            
            # Expand a bit for context
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            fw = min(w - x, fw + 2 * margin)
            fh = min(h - y, fh + 2 * margin)
            
            # Crop and resize (simulate zoom)
            face_crop = frame[y:y+fh, x:x+fw]
            face_crop = cv2.resize(face_crop, (w, h))  # Stretch to fill full screen
            
            # face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            # face_crop = cv2.cvtColor(face_crop, cv2.COLOR_GRAY2BGR)
            # face_crop = cv2.GaussianBlur(face_crop, (5, 5), 2)  
            
            face_crop = enhance_image(face_crop)  # Enhance the image
            
            gaze.refresh(face_crop)
            annotated = gaze.annotated_frame()
            
            # Counter for no gaze recognized using system time
            if not hasattr(main, "no_gaze_start_time"):
                main.no_gaze_start_time = None  # Initialize start time
            
            # Text status
            text = ""
            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"
            
            if text == "":
                if main.no_gaze_start_time is None:
                    main.no_gaze_start_time = time.time()  # Start timing
                elif time.time() - main.no_gaze_start_time > 10:  # Check if 10 seconds passed
                    print("No gaze detected for 10 seconds. Try again.")
                    break
            else:
                main.no_gaze_start_time = None  # Reset start time if gaze is recognized
            
            # Expand a bit for context
            margin = 50
            x = max(0, x - margin)
            y = max(0, y - margin)
            fw = min(w - x, fw + 2 * margin)
            fh = min(h - y, fh + 2 * margin)
            
            # Crop and resize (simulate zoom)
            face_show = frame[y:y+fh, x:x+fw]
            
            cv2.putText(face_show, text, (30, 60), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Gaze Tracker", face_show)
            
        else:
            cv2.putText(frame, "No face detected", (30, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
            cv2.imshow("Gaze Tracker", frame)
        
        if cv2.waitKey(1) == 27:
            break
    
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
