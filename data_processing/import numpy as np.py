import numpy as np
import os
import sys
import pandas as pd
from PIL import Image
import csv

cropped_dir = sys.argv[1]

count = 0


with open('grayscale_data.csv', 'w') as f:
	writer = csv.writer(f)
	for i in os.walk(cropped_dir):
	    for j in np.arange(0,len(i[2])):
	        if str(i[2][j]).endswith('jpg'):
	        	try:
		            file_path = str(i[0]) + "/" +  str(i[2][j])
		            image = Image.open(file_path)
		            #conver to black and white
		            c_pil_image = image.convert('LA')
		            #resize cropped image
		            size = 500,500
		            final = c_pil_image.resize(size)
		            #change image to numpy array
		            pix = np.array(final)
		            pix = pix.flatten()
		            #write row
		            pix = pix.tolist()
		            pix = pix[::2]
		            name = [str(i[2][j])]
		            row = name + pix
		            writer.writerow(row)
		            count += 1
		            if count % 100 == 0:
		            	print(str(count) + ' images converted')
		        except :
		        	print('Image ' + str(i[2][j])+ ' not converted')







