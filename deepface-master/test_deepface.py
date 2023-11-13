from matplotlib import pyplot as plt
from deepface import DeepFace
import os
import glob
import cv2.data

# Specify the path to the directory containing the images
dir_path = "tests/dataset/small_dataset/"

# Get all the image paths
allImages = glob.glob(os.path.join(dir_path, "*.jpg"))

# Array to store all paths to images of 1 person 
Person = []

# Goal is the image of the person we want to find 
imageToFind = "tests/dataset/img1.jpg"

# Iterate over the image paths
for image in allImages:

    # Read the image using cv2
    img = cv2.imread(image)

        
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:

        # Crop the face from the image
        face_crop = img[y:y+h, x:x+w]

        # Face verification for Person
        result = DeepFace.verify(face_crop, imageToFind, enforce_detection=False)
        verified_person = result["verified"]

        # Check which person the face belongs to
        if verified_person:
            print("The face IS from the desired Person.")
            Person.append(image)

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Label the face in the image
            cv2.putText(img, 'Desired Person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        else:
            print("The face does NOT belong to the desired Person.")
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

            # Label the face in the image
            cv2.putText(img, 'Not Desired Person', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # Facial analysis Person
    result = DeepFace.analyze(img, actions=["age", "gender", "race", "emotion"])
    person = result[0]
    print("Age:", person["age"])
    print("Gender:", person["gender"])
    print("Race:", person["dominant_race"])
    print("Emotion:", person["dominant_emotion"])

    #recize the image to fit the screen
    ratio = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * ratio))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    # Display the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Dictionary to store the name of the person in the image
nameOfPerson = {image: 'Angelina Jolie' for image in Person}