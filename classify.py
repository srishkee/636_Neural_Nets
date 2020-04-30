import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
import numpy as np
import random
import glob
import time
from sklearn.utils import class_weight

from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.utils import np_utils
from keras.optimizers import SGD, adam
from keras.layers import *
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras import models, layers

import matplotlib.pyplot as plt

'''
# Note: This file trains a PRETRAINED, INCEPTIONV3 model on the Cooking dataset
Pretrained InceptionV3 network setup from here: https://medium.com/abraia/first-steps-with-transfer-learning-for-custom-image-classification-with-keras-b941601fcad5
'''

print("Running Cutting veggies classifier!")

def get_time():
  ts = time.ctime(time.time()) # Get formatted timestamp
  return ts

def print_training_data(acc, val_acc, loss, val_loss):
  print('Training Accuracy:\n', acc) 
  print('Validation Accuracy:\n', val_acc) 
  print('Training Loss:\n', loss) 
  print('Validation Loss:\n', val_loss) 

def write_data_to_file(acc, val_acc, loss, val_loss):
  # Write data to .csv file
  # Note: Clear data.csv each time! After clearing, add '0' to make it non-empty
  open('data.csv', 'w+').close() # Clear file before writing to it (and create if nonexistent)
  with open('data.csv', 'w') as f:
    f.write('0') # Add a value
  f.close()
  print('Writing data to .csv file...')
  data = pd.read_csv('data.csv', 'w') 
  data.insert(0,"Training Acc", acc)
  data.insert(1,"Training Loss", val_acc)
  data.insert(2,"Validation Acc", loss)
  data.insert(3,"Validation Loss", val_loss)
  data.to_csv('data.csv')
  print('Finished writing data!')

def plot_graphs(acc, val_acc, loss, val_loss):
  # Plot results
  print("Starting plotting...") 
  epochs = range(1, len(acc)+1)
  plt.plot(epochs, acc, 'bo', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and Validation accuracy')
  plt.legend()
  plt.savefig('plots/Cooking_TrainingAcc.png')
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and Validation loss')
  plt.legend()
  plt.ylim([0,4]) # Loss should not increase beyond 4! 
  plt.savefig('plots/Cooking_TrainingLoss.png')
  plt.figure()
  plt.show()
  print('Finished plotting!')


# Construct pretrained InceptionV3 model 
def get_inceptionV3(num_classes, IMG_HEIGHT, IMG_WIDTH):

  # base_model = InceptionV3(weights='imagenet', 
  # include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)) # FIXME: Input_shape!
  # x = base_model.output
  # x = GlobalAveragePooling2D(name='avg_pool')(x)
  # x = Dropout(0.4)(x)
  # x = Dense(num_classes, name='preprediction')(x)
  # predictions = Activation('softmax', name='prediction')(x)
  # model = Model(inputs=base_model.input, outputs = predictions)
  # # Freeze all previous layers 
  # for layer in base_model.layers:
  #   layer.trainable = False

  # Create model
  model = models.Sequential()
  model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH,3)))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(64, (3,3), activation='relu'))
  model.add(layers.MaxPooling2D((2,2)))
  model.add(layers.Conv2D(64, (3,3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(num_classes, activation='softmax'))
  model.summary()
  print("Using MNIST model!")


  # Compile model - default learning rate = 0.001 for RMSprop
  model.compile(loss='categorical_crossentropy', 
    optimizer='rmsprop', 
    metrics=['accuracy'])
  return model

# Load data into 60:20:20 split
def load_data(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, seed=None, root_dir=None):
  if(root_dir==None):
    print("Please enter a valid data directory!")
    return
  
  random.seed(seed)

  def read_from_paths(paths):
    x=[]
    for path in paths:
      # print('path: ', path)
      img = load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
      img = img_to_array(img)
      x.append(img)
    return x

  classes = os.listdir(root_dir)
  classes = sorted(classes)

  train_path = []
  val_path = []
  test_path = []

  train_x, train_y = [],[]
  val_x, val_y = [],[]
  test_x, test_y = [],[]

  # Read paths and split data into 6:2:2 dataset
  for i, _class in enumerate(classes):
    paths = glob.glob(os.path.join(root_dir, _class, "*"))
    paths = [n for n in paths if n.endswith(".JPG") or n.endswith(".jpg")]
    random.shuffle(paths)
    num_plants = len(paths)
    # print("num_plants: ", num_plants)

    train_path.extend(paths[:int(num_plants*0.6)])
    train_y.extend([i]*int(num_plants*0.6))

    val_path.extend(paths[int(num_plants*0.6):int(num_plants*0.8)])
    val_y.extend([i]*len(paths[int(num_plants*0.6):int(num_plants*0.8)]))

    test_path.extend(paths[int(num_plants*0.8):])
    test_y.extend([i]*len(paths[int(num_plants*0.8):]))

  print(get_time(), " Loading images...")

  train_x = read_from_paths(train_path)
  print(get_time(), " Loaded all training images!")
  val_x = read_from_paths(val_path)
  print(get_time(), " Loaded all validation images!")
  test_x = read_from_paths(test_path)
  print(get_time(), " Loaded all testing images!")

  # Convert all to numpy
  train_x = np.array(train_x)/255.
  train_y = np.array(train_y)
  val_x = np.array(val_x)/255.
  val_y = np.array(val_y)
  test_x = np.array(test_x)/255.
  test_y = np.array(test_y)

  # Calculate class weight
  classweight = class_weight.compute_class_weight('balanced', 
    np.unique(train_y), train_y)

  # Convert to categorical (~1-hot encoding)
  train_y = np_utils.to_categorical(train_y, NUM_CLASSES)
  val_y = np_utils.to_categorical(val_y, NUM_CLASSES)
  test_y = np_utils.to_categorical(test_y, NUM_CLASSES)
  print(get_time(), " Successfully loaded data!")

  return train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes

# ------------------------------ Main ------------------------------ #

# Define parameters
NUM_CLASSES = 2
IMG_HEIGHT = 240
IMG_WIDTH = 320
EPOCHS = 4
BATCH_SIZE = 32

# Load model 
model = get_inceptionV3(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH)
print("Successfully loaded model!")

# Load data - will take a while
data_dir = '/scratch/user/skumar55/cooking/Images'
print("Categories: ", os.listdir(data_dir))
train_x, val_x, test_x, train_y, val_y, test_y, classweight, classes = load_data(NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH, seed=7, root_dir=data_dir)

print(train_x.shape, val_x.shape, test_x.shape)   
print(train_y.shape, val_y.shape, test_y.shape)

# Data Augmentation: Shift, zoom, and flip images (horizontally)
train_datagen = ImageDataGenerator(
  rotation_range=10,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.1,
  horizontal_flip=True,
  fill_mode='nearest')

val_datagen = ImageDataGenerator(
  rotation_range=10,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.1,
  horizontal_flip=True,
  fill_mode='nearest')
train_datagen.fit(train_x)
val_datagen.fit(val_x)

# Define callbacks & begin training
es = EarlyStopping(patience=10)
print(get_time(), " Beginning training...")
history = model.fit_generator(train_datagen.flow(train_x, train_y, batch_size=BATCH_SIZE), 
  epochs=EPOCHS, steps_per_epoch=len(train_x)/BATCH_SIZE, 
  validation_data=val_datagen.flow(val_x, val_y, batch_size=BATCH_SIZE), 
  callbacks=[es], class_weight=classweight)

# history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(val_x, val_y), callbacks=[es], class_weight=classweight)

# Save model 
print (get_time(), " Finished training! Saving model...")
model.save('cooking3.h5')

# Save training parameters
print('Obtaining training data...')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
print_training_data(acc, val_acc, loss, val_loss)

# Get testing results 
test_results = model.evaluate(test_x, test_y)
print("Printing test results...")
for metric, name, in zip(test_results, ['loss', 'acc', 'top 5 acc']):
  print(name, metric)

write_data_to_file(acc, val_acc, loss, val_loss)
plot_graphs(acc, val_acc, loss, val_loss)
print("Model finished!")