import cv2
import numpy as np
from datetime import datetime
from tkinter import Tk, Label, Entry, Button
from tkinter import filedialog

# Load pre-trained Haar Cascade classifiers for eye and face detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define color ranges for simulated diagnosis (HSV format)
color_ranges = {
    'blue': ((80, 20, 20), (120, 255, 255)),
    'brown': ((0, 20, 20), (40, 255, 255)),
    'green': ((40, 20, 20), (80, 255, 255)),
    'gray': ((0, 0, 100), (180, 30, 255))
}

# Simulated diagnosis based on eye color
def diagnose_iridology(eye_color):
    if eye_color == 'blue':
        return "Diagnosis: High stress levels and digestive issues"
    elif eye_color == 'brown':
        return "Diagnosis: Liver issues and high toxicity"
    elif eye_color == 'green':
        return "Diagnosis: Mixed type, potential for hormonal imbalance"
    elif eye_color == 'gray':
        return "Diagnosis: Potential for nervous system issues"
    else:
        return "Eye color not recognized."

# Main function for eye detection, iris, pupil, and sclera detection, and diagnosis
def detect_eyes_and_diagnose(frame, name, birth_date):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Region of Interest (ROI) for eye
            eye_roi_bgr = roi_color[ey:ey+eh, ex:ex+ew]
            eye_roi_hsv = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2HSV)
            
            # Perform iridology diagnosis (simulated)
            mean_color = cv2.mean(eye_roi_hsv)
            eye_color = get_eye_color(mean_color[:3])
            diagnosis = diagnose_iridology(eye_color)
            
            # Print diagnosis to console
            print(diagnosis)
            
            # Draw diagnosis text near the eye
            cv2.putText(frame, diagnosis, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Detect iris, pupil, and sclera
            iris, pupil, sclera = detect_iris_pupil_sclera(eye_roi_bgr)
            
            # Mark and check color of iris, pupil, and sclera
            if iris:
                iris_color = check_color(eye_roi_bgr, iris)
                cv2.rectangle(roi_color, (ex + iris[0], ey + iris[1]), (ex + iris[0] + iris[2], ey + iris[1] + iris[3]), (255, 0, 0), 2)
                cv2.putText(frame, f"Iris: {iris_color}", (x + ex + iris[0], y + ey + iris[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            if pupil:
                pupil_color = check_color(eye_roi_bgr, pupil)
                cv2.rectangle(roi_color, (ex + pupil[0], ey + pupil[1]), (ex + pupil[0] + pupil[2], ey + pupil[1] + pupil[3]), (0, 0, 255), 2)
                cv2.putText(frame, f"Pupil: {pupil_color}", (x + ex + pupil[0], y + ey + pupil[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if sclera:
                sclera_color = check_color(eye_roi_bgr, sclera)
                cv2.rectangle(roi_color, (ex + sclera[0], ey + sclera[1]), (ex + sclera[0] + sclera[2], ey + sclera[1] + sclera[3]), (0, 255, 255), 2)
                cv2.putText(frame, f"Sclera: {sclera_color}", (x + ex + sclera[0], y + ey + sclera[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Add user details to frame
    user_details = f"Name: {name}, Birth Date: {birth_date}, Date: {datetime.now().strftime('%Y-%m-%d')}"
    cv2.putText(frame, user_details, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# Determine eye color based on HSV values
def get_eye_color(hsv_color):
    for color, (lower, upper) in color_ranges.items():
        if np.all(hsv_color >= lower) and np.all(hsv_color <= upper):
            return color
    return 'unknown'

# Function to detect iris, pupil, and sclera
def detect_iris_pupil_sclera(eye_roi_bgr):
    gray = cv2.cvtColor(eye_roi_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    iris = None
    pupil = None
    sclera = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            if 0.7 <= aspect_ratio <= 1.3:
                pupil = (x, y, w, h)
            elif aspect_ratio > 1.3:
                iris = (x, y, w, h)
            elif aspect_ratio < 0.7:
                sclera = (x, y, w, h)
    
    return iris, pupil, sclera

# Function to check the dominant color in a given ROI
def check_color(roi_bgr, region):
    x, y, w, h = region
    roi = roi_bgr[y:y+h, x:x+w]
    
    # Convert ROI to HSV color space
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram in HSV color space
    hist_hsv = cv2.calcHist([roi_hsv], [0], None, [180], [
                             0, 180])

    # Get the color with maximum frequency
    color = np.argmax(hist_hsv)
    return color


def save_frame():
    # Check if there is a processed frame
    if processed_frame is not None:
        # Get the file path to save the image
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg")])
        # If a file path was selected, save the image
        if file_path:
            cv2.imwrite(file_path, cv2.cvtColor(
                processed_frame, cv2.COLOR_RGB2BGR))


def start_capture():
    # Create a video capture object
    cap = cv2.VideoCapture(0)

    # Check if the capture object is opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Create a window named 'Eye Detection'
    cv2.namedWindow('Eye Detection', cv2.WINDOW_NORMAL)

    # Start reading and displaying frames
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Process the frame: detect eyes and perform diagnosis
        global processed_frame
        processed_frame = detect_eyes_and_diagnose(
            frame, entry_name.get(), entry_date.get())

        # Display the processed frame
        cv2.imshow('Eye Detection', processed_frame)

        # Check for user input to quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


# Create a tkinter window
root = Tk()
root.title("Eye Detection and Diagnosis")

# Label and Entry for Name
label_name = Label(root, text="Name:")
label_name.grid(row=0, column=0, padx=10, pady=10)
entry_name = Entry(root)
entry_name.grid(row=0, column=1, padx=10, pady=10)

# Label and Entry for Birth Date
label_date = Label(root, text="Birth Date:")
label_date.grid(row=1, column=0, padx=10, pady=10)
entry_date = Entry(root)
entry_date.grid(row=1, column=1, padx=10, pady=10)

# Button to start capturing
button_start = Button(root, text="Start Capture", command=start_capture)
button_start.grid(row=2, column=0, padx=10, pady=10)

# Button to save the processed frame
button_save = Button(root, text="Save Frame", command=save_frame)
button_save.grid(row=2, column=1, padx=10, pady=10)

# Start the tkinter main loop
root.mainloop()
