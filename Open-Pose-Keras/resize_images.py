import cv2
import os

print(cv2.__version__)

'''
This script resizes all images to (RESIZE_WIDTH, RESIZE_HEIGHT)
'''

RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

input_folder_name = 'NotCutting_Original/'

imgs = os.listdir(input_folder_name)
for (i,filename) in enumerate(imgs):
	# if(i == 5): break
	print("filename: ", filename)

	# Read image
	input_img_name = input_folder_name + filename
	img = cv2.imread(input_img_name)

	# Resize image
	resized_img = cv2.resize(img, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation = cv2.INTER_LINEAR) # Resize to match video size of (480x640)

	# Save resized image
	output_img_name = 'NotCutting/' + filename
	cv2.imwrite(output_img_name, resized_img) # Write frame to Images/


print("Resized", len(imgs), "images in", input_folder_name, "to (", RESIZE_WIDTH, "x", RESIZE_HEIGHT, ") !")