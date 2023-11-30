import numpy as np
import os
import cv2
import sys

# Define a dictionary to map numerical labels to emotions
emotion_mapping = {
    0: "Neutral",
    1: "Happy",
    2: "Sad"
}

def read_images(path, sz=None):
    c = 0
    X, y = [], []

    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            label = int(subdirname)  # Assuming subdirectory names are integers representing mood labels
            for filename in os.listdir(subject_path):
                try:
                    if filename == ".directory":
                        continue
                    filepath = os.path.join(subject_path, filename)
                    
                    # Check if the file has a .pgm extension
                    if filepath.endswith('.png'):
                        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                        
                        # Resize the images to the prescribed size
                        if sz is not None:
                            im = cv2.resize(im, (200, 200))
                            
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(label)

                except IOError as e:
                    print(f"I/O Error({e.errno}): {e.strerror}")
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise
            c += 1

    print(f"Loaded {len(X)} images for training.")
    return [X, y]

# Path to your datasets
pathing = r"C:/Users/Rein/OneDrive/Desktop/Midterm_exam/Face Recognition/Face Recognition/Emotions"

# Read .png images and labels from the specified directory
[X, y] = read_images(pathing)

def mood_detection(X, y):
    # Create and train the face recognition model
    model = cv2.face.EigenFaceRecognizer_create()  # You can choose other models here
    model.train(np.asarray(X), np.asarray(y))

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Create a directory to save captured images
    save_path = 'Captured_Images'
    os.makedirs(save_path, exist_ok=True)
    img_counter = 0

    while True:
        ret, img = camera.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                label, confidence = model.predict(roi)
                predicted_emotion = emotion_mapping.get(label, "Unknown")
                cv2.putText(img, "Emotion: " + predicted_emotion, (x, y - 20), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
            except:
                continue

            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)

        cv2.imshow("Mood Detection", img)
        key = cv2.waitKey(1)

        # Capture and save image on mouse click (change 'c' to any key of your choice)
        if key & 0xFF == ord("c"):
            img_name = os.path.join(save_path, f"emotions{img_counter}.png")
            cv2.imwrite(img_name, img)
            print(f"{img_name} captured")
            img_counter += 1

        if key & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mood_detection(X, y)
