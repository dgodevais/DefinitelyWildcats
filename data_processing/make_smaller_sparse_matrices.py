import os
import sys
import scipy.io
import numpy as np
import pandas as pd
import dlib
import face_recognition as fr
from PIL import Image, ImageFile
from operator import itemgetter
from scipy.sparse import *

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
'''
Run script with:
python3 make_smaller_sparse_matrices.py unpacked_tar_dir mat_file group output_file
i.e:
python3 make_smaller_sparse_matrices.py imdb_1 imdb.mat imdb output_file_1.npz
'''

unpacked_tar_dir = sys.argv[1]
mat_file = sys.argv[2]
group = sys.argv[3]
output_file = sys.argv[4]

# imdb mat file
label_data = scipy.io.loadmat(mat_file)
gender = label_data[group]['gender']
file_loc = label_data[group]['full_path']
dob =  label_data[group]['dob']
taken_year = label_data[group]['photo_taken']
df = pd.DataFrame({'file':file_loc[0][0][0], 'gender':gender[0][0][0], 
                   'dob': dob[0][0][0]/365, 'taken_year': taken_year[0][0][0], })#, index=range(0,len(file)))
df['age'] = df['taken_year'] - df['dob']
df['file'] = df['file'].map(lambda x: x[0].lstrip('[]').split('/', 1)[1])

size = 100,100
matrix_count = 1
image_count = 0
image_list = []
age_list = []
gender_list = []
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
            try:
                image = fr.load_image_file(image_loc)
                face_locations = fr.face_locations(image)
            except:
            	print(str(i[2][j]) + " failed to load")
            	continue
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
            image_list.append(i[2][j])
            age_list.append(float(df[df['file'] == str(i[2][j])]['age']))
            gender_list.append(float(df[df['file'] == str(i[2][j])]['gender']))
            image_count += 1
            sparse_image_count = csr_matrix((np.arange(len(image_list)),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
                    shape=(len(image_list), 1))
            if image_count % 1000 == 0 :
                print('Saving matrix #' + str(matrix_count))
                sparse_image_count = csr_matrix((np.arange(len(image_list)),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
                    shape=(len(image_list), 1))
                sparse_age = csr_matrix((np.asarray(age_list),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
                    shape=(len(image_list), 1))
                sparse_gender = csr_matrix((np.asarray(gender_list),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
                    shape=(len(image_list), 1))
                sparse_matrix = hstack([sparse_image_count, sparse_matrix, sparse_age, sparse_gender])
                thousand_output_file = str(matrix_count) + "_" + output_file
                save_npz(thousand_output_file, sparse_matrix)   
                image_count = 0
                image_list = []
                age_list = []
                gender_list = []
                matrix_count += 1
print('Saving matrix #' + str(matrix_count))
sparse_image_count = csr_matrix((np.arange(len(image_list)),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
    shape=(len(image_list), 1))
sparse_age = csr_matrix((np.asarray(age_list),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
    shape=(len(image_list), 1))
sparse_gender = csr_matrix((np.asarray(gender_list),(np.arange(len(image_list)), np.repeat(0, len(image_list)))),\
    shape=(len(image_list), 1))
sparse_matrix = hstack([sparse_image_count, sparse_matrix, sparse_age, sparse_gender])
thousand_output_file = str(matrix_count) + "_" + output_file
save_npz(thousand_output_file, sparse_matrix)   

