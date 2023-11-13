from deepface import DeepFace
import matplotlib.pyplot as plt
import os
import glob
from PIL import Image
import cv2.data
from mtcnn import MTCNN

##### Start of: Format check #####
def convert_to_jpg(image_path):
    img = Image.open(image_path)
    converted_path = os.path.splitext(image_path)[0] + '.jpg'
    img.save(converted_path, 'JPEG')
    os.remove(image_path)    #delete old image with wrong format
    return converted_path

# Specify the path to the directory containing the images
dir_path = "tests/dataset/small_dataset/"

allimage_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff', '*.ico', '*.jfif', '*.webp']
wrong_formats = ['.jpeg', '.png', '.bmp', '.gif', '.tiff', '.ico', '.jfif', '.webp']
allImages = []
for image_format in allimage_formats:
    allImages.extend(glob.glob(os.path.join(dir_path, image_format)))

# Convert all images to .jpg format
convertedImages = []
imagesToDelete = []
for image_path in allImages:
    if os.path.splitext(image_path)[1].lower() in wrong_formats:
        imagesToDelete.append(image_path) #add to list of images with wrong format to delete
        converted_path = convert_to_jpg(image_path) #convert to jpg
        convertedImages.append(converted_path) #add to list of converted images
for image in imagesToDelete:
    allImages.remove(image) #remove from the list allImages

allImages.extend(convertedImages)  #join the list of converted images to the list allImages
del convertedImages #delete convertedImages

##### End of: Format Check #####

# Array to store all paths to images of 1 person 
Person = []

# Goal is the image of the person we want to find 
imageToFind = "tests/dataset/img1.jpg"
#check if imageToFind is with the correct format
if os.path.splitext(imageToFind)[1].lower() in wrong_formats:
    imageToFind = convert_to_jpg(imageToFind)

# Iterate over the image paths
for image in allImages:
    print("\nImage: ", image)
    # Face verification for Person
    result = DeepFace.verify(image, imageToFind, enforce_detection=False)
    verified_person = result["verified"]

    # Check which person the face belongs to
    if verified_person:
        print("The face IS from the desired Person.")
        Person.append(image)
    else:
        print("The face does NOT belong to the desired Person.")

    # Facial analysis Person
    # result = DeepFace.analyze(image, actions=["age", "gender", "race", "emotion"])
    # person = result[0]
    # print("Age:", person["age"])
    # print("Gender:", person["gender"])
    # print("Race:", person["dominant_race"])
    # print("Emotion:", person["dominant_emotion"])


# Dictionary to store the name of the person in the image
# nameOfPerson = {image: 'Angelina Jolie' for image in Person}
print("Person list: ",Person)
# Print how many photos were found of the person
print("Images that have the correct face: ", len(Person))

# Iterate over the image paths
# for img_path in Person:
#     # Extract faces
#     detected_faces = DeepFace.extract_faces(img_path, enforce_detection = False)

#     # Display the faces if any were detected
#     if detected_faces:
#         # Iterate over the detected faces
#         for face_obj in detected_faces:
#             face = face_obj["face"]
#             plt.imshow(face)  # Display the face image
#             plt.title(f'{nameOfPerson[img_path]}')  # Label the face
#             plt.show()
#     else:
#         print("No faces detected in the image.")