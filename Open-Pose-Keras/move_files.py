import os
import random
import shutil

'''
This script moves file from one directory to another
Taken from: https://stackoverflow.com/questions/29705305/python-move-10-files-at-a-time

Files processed:
[ climb, dribble, golf, kick_ball, pullup, push, pushup, situp, shoot_bow, swing_baseball ]
'''

FOLDER_NAME = "shoot_bow"

source = "Images_Cutting_Folders/" + FOLDER_NAME
dest = "NotCutting_Original/"

files = os.listdir(source) # Get all files in folder
for filename in random.sample(files, min(len(files), 200)): # Randomly sample 200 files
	print("filename: ", filename)
	path = os.path.join(source, filename)
	shutil.move(path, dest) # Move files


