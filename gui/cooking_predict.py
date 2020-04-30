import numpy as np
from keras import layers
import keras.backend as K
from keras.models import Model, load_model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import load_img, img_to_array

def get_image(img_path, IMG_HEIGHT, IMG_WIDTH):
	img = load_img(img_path, target_size=(IMG_HEIGHT,IMG_WIDTH))
	img = img_to_array(img)/255.
	return img

# -------------------------------------- main (testing) -------------------------------------- # 
# Use relative paths!
# weights_path = "Documents/toda_okura_inceptionv3.h5"
# img_path = "Desktop/Grape__Esca__Black_Measles.jfif" 

# model = load_model(weights_path)
# test_image = get_image(img_path)
# predictions = np.argmax(model.predict(test_image[np.newaxis,:]))
# print("predictions: ", predictions)