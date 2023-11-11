from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import glob

# Specify the path to the directory containing the images
dir_path = "tests/dataset/small_dataset/"

# Get all the image paths
image_paths = glob.glob(os.path.join(dir_path, "*.jpg"))

#array to store all paths to images of 1 person 
Jolie = []

#goal is the image of the person we want to find 
goal = "tests/dataset/img1.jpg"

# Iterate over the image paths
for img_path in image_paths:
    # Face verification
    result = DeepFace.verify(img_path, goal) #goal is the image of the person we want to find 
    verified = result["verified"]
    print("Is verified:", verified)
    if verified:
        Jolie.append(img_path)

    # Facial analysis
    result = DeepFace.analyze(img_path, actions=["age", "gender", "race", "emotion"])
    person = result[0]
    print("Age:", person["age"])
    print("Gender:", person["gender"])
    print("Race:", person["dominant_race"])
    print("Emotion:", person["dominant_emotion"])

#print how many photos were found of the person
print("Images that have the correct face: ", len(Jolie))
# Iterate over the image paths
for img_path in Jolie:
    # Extract faces
    detected_faces = DeepFace.extract_faces(img_path, enforce_detection=True)

    # Display the faces if any were detected
    if detected_faces:
        # Iterate over the detected faces
        for face_obj in detected_faces:
            face = face_obj["face"]
            plt.imshow(face)  # Display the face image
            plt.show()
    else:
        print("No faces detected in the image.")