# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np

# Create a directory to save the images
output_dir = 'Emotions'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the pre-trained face detection classifier (Haar Cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the video file path

# Counter for saved images
image_count = 0

# Define the size for the captured face images
image_size = (200, 200)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture video feed")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Crop and resize the detected face
        face = cv2.resize(gray[y:y + h, x:x + w], image_size)

        # Check if the w is pressed and save the face image in .png format
        if cv2.waitKey(1) & 0xFF == ord('w'):
            image_count += 1
            image_name = os.path.join(output_dir, f"face_{image_count}.png")
            cv2.imwrite(image_name, face)

        # Display the detected face with a rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Face Detection', frame)

    # Exit the loop on pressing the 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
