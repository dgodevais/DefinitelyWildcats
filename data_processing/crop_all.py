import os
import sys
import numpy as np
import dlib
import face_recognition as fr
from PIL import Image

unpacked_tar_dir = sys.argv[1]
cropped_dir = sys.argv[2]

detected_count = 0
not_detected_count = 0

for i in os.walk(unpacked_tar_dir):
    # i[0] is dir
    # i[2] is list of files
    for j in np.arange(0,len(i[2])):
        if str(i[2][j]).endswith('jpg'):
            image_loc = i[0] + "/" + i[2][j]
            image = fr.load_image_file(image_loc)
            face_locations = fr.face_locations(image)
            # try for 1st face
            try:
                # find face location
                face_location = face_locations[0]
                top, right, bottom, left = face_location
                # crop original image
                face_image = image[top:bottom, left:right]
                pil_image = Image.fromarray(face_image)
                # save cropped image with new name
                img_fname = "cropped_" + i[2][j]
                pil_image.save(cropped_dir + img_fname)
                detected_count += 1
            except:
                #print("In image" + i[2][j] + " no faces were detected")
                not_detected_count += 1
        if detected_count % 100 == 0:
        	print(str(detected_count) + " images with face detected")
print(str(detected_count) + " images with face detected")
print(str(not_detected_count) + " images with no face detected")