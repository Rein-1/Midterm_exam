# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 05:36:57 2023

@author: Rein
"""
import cv2
import os

# Load pre-trained FER model for emotion detection
emotion_classifier = cv2.face.createFacemarkLBF()
emotion_classifier.loadModel(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Create a directory to save images
save_directory = "detected_emotions"
os.makedirs(save_directory, exist_ok=True)

image_count = 0

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the pre-trained FER model
    faces = emotion_classifier.fit(gray)

    # Display the detected faces with emotions
    for face in faces[1]:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the region of interest (ROI) for emotion detection
        roi_gray = gray[y:y+h, x:x+w]

        # Perform emotion detection on the ROI
        # (This part needs improvement by training a more robust model)
        detected_emotion = "Happy"
        cv2.putText(frame, f"Emotion: {detected_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the frame when an emotion is detected
        if detected_emotion != "Unknown":
            image_count += 1
            filename = f"detected_emotion_{image_count}.jpg"
            cv2.imwrite(os.path.join(save_directory, filename), frame)
            print(f"Emotion detected and saved as {filename}")

    # Display the resulting frame
    cv2.imshow('Mood Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video_capture.release()
cv2.destroyAllWindows()

