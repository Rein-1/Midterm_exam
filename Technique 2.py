import cv2

# Load reference images for happy, sad, and neutral emotions
happy_img = cv2.imread('Happy.png', 0)  # Load in grayscale
sad_img = cv2.imread('Sad.png', 0)
neutral_img = cv2.imread('Neutral.png', 0)

# Initialize ORB
orb = cv2.ORB_create()

# Find keypoints and descriptors in the reference images
kp_happy, des_happy = orb.detectAndCompute(happy_img, None)
kp_sad, des_sad = orb.detectAndCompute(sad_img, None)
kp_neutral, des_neutral = orb.detectAndCompute(neutral_img, None)

# Load pre-trained Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from the camera
video_capture = cv2.VideoCapture(0)  # Use 0 for the default webcam, or provide the video file path

while True:
    ret, frame = video_capture.read()  # Read a frame from the video feed

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    emotion = "Unknown"  # Initialize emotion as Unknown

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # Crop face region for emotion detection
        face_roi = gray[y:y+h, x:x+w]

        # Detect keypoints and descriptors in the face region
        kp_frame, des_frame = orb.detectAndCompute(face_roi, None)

        # Initialize BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Check if descriptors exist
        if des_frame is not None:
            # Match descriptors between face region and reference images
            matches_happy = bf.match(des_happy, des_frame) if des_happy is not None else []
            matches_sad = bf.match(des_sad, des_frame) if des_sad is not None else []
            matches_neutral = bf.match(des_neutral, des_frame) if des_neutral is not None else []

            threshold = 10  # Define an arbitrary threshold

            if len(matches_happy) > threshold:
                emotion = "Happy"
            elif len(matches_sad) > threshold:
                emotion = "Sad"
            elif len(matches_neutral) > threshold:
                emotion = "Neutral"

        # Display the detected emotion label
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the processed frame with bounding boxes and emotion label
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the OpenCV windows
video_capture.release()
cv2.destroyAllWindows()
