import cv2
import csv

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start video capture ('0' for the default camera)
video_capture = cv2.VideoCapture(0)

count = 0
image_paths = []

while count < 10:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display output
    cv2.imshow('Video', frame)

    # Save 10 images in PGM format
    if len(faces) > 0:
        count += 1
        filename = f'image_{count}.pgm'
        cv2.imwrite(filename, gray[y:y+h, x:x+w])
        image_paths.append(filename)

    # Stop if Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture
video_capture.release()
cv2.destroyAllWindows()

# Write image paths to CSV
with open('image_paths.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image Paths'])
    for path in image_paths:
        writer.writerow([path])
