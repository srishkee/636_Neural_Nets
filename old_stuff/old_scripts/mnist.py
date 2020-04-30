from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np

# This code sample creates a neural network to classify digits based on the MNIST dataset 

# Get MNIST training + testing data (part of Keras)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Create & Compile model
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax', name='main_output'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Parse data into correct formation 
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
print("train_images: ", train_images.shape)

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255
print("test_images: ", test_images.shape)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Fit module to data (runs 5 epochs on training data!)
model.fit(train_images, train_labels, epochs=1, batch_size=128)

# model.print("Saving model...")
# model.save('mnist_model.h5')

# Evaluate model on test data! 
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print ('test_loss', test_loss)
# print ('test_acc', test_acc)



predictions = model.predict(test_images)

# predictions = np.array([[1,2,3,4,5,6,7,8,9,0], [1,2,3,4,5,6,7,8,9,0]])

print("Printing predictions...")
for (i,pred) in enumerate(predictions):
	print(pred)
	max_val = np.max(pred)
	max_idx = np.argmax(pred)
	print('Prediction: Number = ', max_idx, ' with accuracy = ', max_val*100, '%\n')
	if(i==5): 
		break