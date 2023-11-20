import glob
import os
import subprocess
import time
import shutil


# Path to your test_deepface.py script
script_path = "test_deepface.py"

# Directory path
dir_path = "tests/dataset/small_dataset/"

##### COMMENT THOSE THAT ARE NOT BEEING TESTED #####
# List of images to convert: ALL JPG
images = ["img2.jpg", "brad.jpg", "Brad_Pitt_Fury_2014.jpg", "couple.jpg", "img1.jpg", "img3.jpg", "img11.jpg", "img14.jpg", "img55.jpg", "img57.jpg"] 
benchmark_path = "tests/dataset/benchmark_alljpg/"

# List of images to convert: half jpg & half png 
# images = ["img2.png", "brad.png", "Brad_Pitt_Fury_2014.png", "couple.png", "img1.png", "img3.jpg", "img11.jpg", "img14.jpg", "img55.jpg", "img57.jpg"] 
# benchmark_path = "tests/dataset/benchmark_halfpng/"

# List of images to convert: all png
# images = ["img2.png", "brad.png", "Brad_Pitt_Fury_2014.png", "couple.png", "img1.png", "img3.png", "img11.png", "img14.png", "img55.png", "img57.png"] 
# benchmark_path = "tests/dataset/benchmark_allpng/"

######################################################


# Run the script 5 times
average=[]
for i in range(5):
    # Delete all images in the directory
    files = glob.glob(dir_path + '*')
    for f in files:
        os.remove(f)
    # Restore the original images
    for image in images: 
        shutil.copy(benchmark_path +image , dir_path+ image)

    start_time = time.time()
    subprocess.run(["python", script_path])
    end_time = time.time()
    elapsed_time = round(end_time - start_time -6.5, 3) #-6.5 is the average time it takes to load the model in my computer
    print(f"Run {i+1}, Elapsed time: {elapsed_time} seconds")
    average.append(elapsed_time)

    
    

print(f"Average time: {round(sum(average)/len(average), 2)} seconds")