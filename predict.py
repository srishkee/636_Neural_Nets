import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
import numpy as np
import random
import glob
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils
from keras.models import load_model

# Get testing results 

# Create graphs
def plot_graphs(INPUT_FOLDER, x, y):
  # Plot results
  print("Starting plotting...") 
  time = range(1, len(x)+1)
  plt.plot(time, y)
  plt.xticks(np.arange(0, len(time), step=5))  # Set label locations.
  plt.xlabel('Seconds')
  plt.ylim([-0.1,1.1]) 
  plt.title('Prediction Results')
  plt.savefig(INPUT_FOLDER + 'video_results.png')
  print('Finished plotting!')

# Create JSON file
def write_to_file(INPUT_FOLDER, final_results):
	f = open(INPUT_FOLDER + 'timeLabel.json', 'w+')
	f.write("{\"Cutting\":")
	f.write(str(final_results))
	f.write("}")
	f.close()

def load_test_data(IMG_HEIGHT, IMG_WIDTH, data_dir):

	def read_from_paths(paths):
	  x=[]
	  for i, path in enumerate(paths):
	    img = load_img(path, target_size=(IMG_HEIGHT,IMG_WIDTH))
	    img = img_to_array(img)
	    x.append(img)
	  return x

	paths = os.listdir(data_dir)

	# REMOVE!!!
	# Choose certain filenames
	# paths1 = []
	# for p in paths:
	# 	if("video" not in p): paths1.append(p)
	# paths = paths1

	# Or, choose an individual frame
	# paths = ['fridge3.jpg']

	for i in range(len(paths)): paths[i] = data_dir + paths[i] # Append folder (data_dir) name
	# random.shuffle(paths)
	paths.sort()
	num_imgs = len(paths)
	# print("paths:", paths)

	print("Loading testing data...")
	pred_x = []
	pred_x = read_from_paths(paths)
	print("Loaded all testing images!")

	pred_x = np.array(pred_x)/255.
	print("Successfully loaded", len(pred_x), "test images!")
	return paths, pred_x

# ------------------------------ Main ------------------------------ #

# Only edit these parameters!
IMG_HEIGHT = 240
IMG_WIDTH = 320
INPUT_FOLDER = 'testing_data/test_vid_4/test_imgs/'
print("INPUT_FOLDER: ", INPUT_FOLDER)

# DON'T EDIT this part!
paths, pred_x = load_test_data(IMG_HEIGHT, IMG_WIDTH, INPUT_FOLDER) # CHANGE FOLDER LATER!!! 
model = load_model("cooking3.h5")

print("Predicting...")
predictions = model.predict(pred_x, batch_size=32)
print("len(predictions): ", len(predictions))

classification_list = [0] * len(predictions)
final_results = [[]] * len(predictions)
for (i,pred) in enumerate(predictions):
  
  # Get frame index (since paths contains a random assortment of frames)
  # idxs = re.findall(r'\d+', paths[i])
  # idx = int(idxs[-1])

  if(pred[0] > pred[1]): # If cutting
  	type = 'Cutting'
  # 	classification_list[idx] = 0
  else: 
  	type = 'NotCutting'
  # 	classification_list[idx] = 1

  print(paths[i], pred, "->", type)
  # final_results[idx] = [idx, pred[0]]


# final_results.sort()
# print("final_results: ", final_results)
# print("classification_list: ", classification_list)

# time = range(len(classification_list))
# plot_graphs(INPUT_FOLDER, time, classification_list)
# write_to_file(INPUT_FOLDER, final_results)
