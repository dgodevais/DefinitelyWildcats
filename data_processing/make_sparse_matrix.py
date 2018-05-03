import os
import sys
import numpy as np
import pandas as pd
import dlib
import face_recognition as fr
from PIL import Image
from operator import itemgetter
from scipy.sparse import *

unpacked_tar_dir = sys.argv[1]

size = 100,100
sparse_dict = {}
image_count = 0
image_list = []
row_list = []
width = 100
keys = ['chin',
'left_eyebrow',
'right_eyebrow',
'nose_bridge',
'nose_tip',
'left_eye',
'right_eye',
'top_lip',
'bottom_lip']

for i in os.walk(unpacked_tar_dir):
    # i[0] is dir
    # i[2] is list of files
    for j in np.arange(0,len(i[2])):
        if str(i[2][j]).endswith('jpg'):
            image_loc = i[0] + "/" + i[2][j]
            image = fr.load_image_file(image_loc)
            face_locations = fr.face_locations(image)
        try:
            face_location = face_locations[0]
        except:
            continue
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        c_pil_image = pil_image.convert('L')
        #resize cropped image
        small = c_pil_image.resize(size)
        small_array = np.array(small)
        try:
            feature_dict = fr.face_landmarks(small_array)[0]
        except:
            continue
        count = 0
        dense_array = np.array([-1])
        if bool(feature_dict):
            for c, k in enumerate(keys):
                if k in feature_dict:
                    feature_dict[k] = [t for t in feature_dict[k] if (max(t) < 100) and (min(t) >= 0)]
                    if len(feature_dict[k]) > 0:
                        zipped = list(zip(*feature_dict[k]))
                        x = zipped[0]
                        y = zipped[1]
                        feature_indexes = np.array(y) * width + np.array(x) + (c * 10000)
                        dense_array = np.append(dense_array, feature_indexes)
                        count += 1
                    else :
                        continue
            dense_array = np.delete(dense_array, 0)
            dense_array = np.sort(dense_array)
            sparse_row = csr_matrix((np.ones(len(dense_array)),(np.repeat(0, len(dense_array)), dense_array)),\
                                    shape=(1, (width ** 2) * len(keys)))
            if image_count == 0 :
                sparse_matrix = sparse_row
            else :
                sparse_matrix = vstack([sparse_matrix, sparse_row])
            row_list.append(image_count)
            image_list.append(i[2][j])
            image_count += 1
spare_image_count = csr_matrix((np.arange(len(image_list)),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
    shape=(len(image_list), 1))
sparse_matrix = hstack([spare_image_count, sparse_matrix])
processed_files_df = pd.DataFrame({'file_name':image_list, 'image_number':row_list})
processed_files_df.to_csv('processed_filenames.csv')
save_npz('sparse_matrix.npz', sparse_matrix)