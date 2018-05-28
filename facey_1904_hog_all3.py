
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import face_recognition
import os
import pandas as pd
import sys
import dlib
from skimage import io
import openface


predictor_model = "shape_predictor_68_face_landmarks.dat"

path = r'images'
image_path = [os.path.join(path, f) for f in os.listdir(path)]
image_path.sort()

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = openface.AlignDlib(predictor_model)

#win = dlib.image_window()

for i in range(len(image_path)):
    image = io.imread(image_path[i])
    detected_faces = face_detector(image, 1)
    if len(detected_faces) == 1:
        for j, face_rect in enumerate(detected_faces):
            pose_landmarks = face_pose_predictor(image, face_rect)
    
            alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    
    #plt.imshow(alignedFace)

	# Save the aligned image to a file
            cv2.imwrite("refined_images/"+image_path[i][18:25]+str(i)+".jpg", alignedFace)
    


####Above we wrote refined_images

image = io.imread(image_path[0])
detected_faces = face_detector(image, 1)


print("I found {} faces in the file {}".format(len(detected_faces), image_path[10]))

# Open a window on the desktop showing the image
#win.set_image(image)

for i, face_rect in enumerate(detected_faces):

	# Detected faces are returned as an object with the coordinates 
	# of the top, left, right and bottom edges
    print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Draw a box around each face we found
    #win.add_overlay(face_rect)

	# Get the the face's pose
    pose_landmarks = face_pose_predictor(image, face_rect)
    
    alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    
    #plt.imshow(alignedFace)

	# Save the aligned image to a file
    cv2.imwrite("refined_images/"+image_path[8][18:25]+str(i)+".jpg", alignedFace)

	# Draw the face landmarks on the screen.
	#win.add_overlay(pose_landmarks)
	        
dlib.hit_enter_to_continue()




imgplot = plt.imshow(image)
plt.show()


