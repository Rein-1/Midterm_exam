import numpy as np
import cv2
import pandas as pd

def read_images(path):
    c = 0
    i = 0
    X, y = [], []
    df = pd.read_csv(path)
    paths = df.iloc[:, 0]

    while (paths.size > i):
        filepath = str(paths[i])
        im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        if im is not None:
            # Resize the images to the prescribed size
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(c)
        else:
            print(f"Warning: Unable to read image at {filepath}")

        i += 1
        c += 1

    return [X, y]

def face_rec():
    names = ['Mercado, Mark Renier R.']  # Put your names here for faces to recognize
    [X, y] = read_images('FACES.csv')
    if len(X) == 0:
        print("No data found. Please check if the file paths in the CSV file are correct.")
        return

    y = np.asarray(y, dtype=np.int32)
    model = cv2.face_LBPHFaceRecognizer.create()
    model.train(X, y)

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, img = camera.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)
            params = model.predict(roi)
            print(params)

            if 0 <= params[0] < 100 and params[1] < 500:
                cv2.putText(img, names[0] + ", " + str(params[1]), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            else:  # Stranger
                cv2.putText(img, 'Not Recognized' + ", " + str(params[1]), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Change the rectangle color to red

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()
