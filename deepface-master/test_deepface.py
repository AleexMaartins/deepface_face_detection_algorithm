from matplotlib import pyplot as plt
from deepface import DeepFace
import os
import glob
import cv2

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

    # Face verification for Person
    result = DeepFace.verify(image, imageToFind)
    verified_person = result["verified"]

    # Check which person the face belongs to

    if verified_person:
        print("The face IS from the desired Person.")
        Person.append(image)
    else:
        print("The face does NOT belong to the desired Person.")
        

    # Facial analysis Person
    result = DeepFace.analyze(image, actions=["age", "gender", "race", "emotion"])
    person = result[0]
    print("Age:", person["age"])
    print("Gender:", person["gender"])
    print("Race:", person["dominant_race"])
    print("Emotion:", person["dominant_emotion"])

# Dictionary to store the name of the person in the image
nameOfPerson = {image: 'Angelina Jolie' for image in Person}

# Print how many photos were found of the person
print("Images that have the correct face: ", len(Person))


#Iterate over the image paths
for img_path in Person:
    # Extract faces
    detected_faces = DeepFace.extract_faces(img_path, enforce_detection=True)

    # Display the faces if any were detected
    if detected_faces:
        # Iterate over the detected faces
        for face_obj in detected_faces:
            face = face_obj["face"]
            plt.imshow(face)  # Display the face image
            plt.title(f'{nameOfPerson[img_path]}')  # Label the face
            plt.show()
    else:
        print("No faces detected in the image.")