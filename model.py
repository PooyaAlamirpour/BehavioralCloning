import os
import matplotlib.image as mpimg
import numpy as np
import cv2
import pandas
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
import math

# Loading the labels file(*.csv)
colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
data = pandas.read_csv('new_behavioral_cloning.csv', skiprows=[0], names=colnames)
center = data.center.tolist()
center_recover = data.center.tolist() 
left = data.left.tolist()
right = data.right.tolist()
steering = data.steering.tolist()
steering_recover = data.steering.tolist()

#  Shuffle the dataset
center, steering = shuffle(center, steering)
center, X_valid, steering, y_valid = train_test_split(center, steering, test_size = 0.10, random_state = 100) 

d_straight, d_left, d_right = [], [], []
a_straight, a_left, a_right = [], [], []

# Make restriction for steering wheel
for i in steering:
  index = steering.index(i)
  if i > 0.15:
    d_right.append(center[index])
    a_right.append(i)
  if i < -0.15:
    d_left.append(center[index])
    a_left.append(i)
  else:
    d_straight.append(center[index])
    a_straight.append(i)

ds_size, dl_size, dr_size = len(d_straight), len(d_left), len(d_right)
main_size = math.ceil(len(center_recover))
l_xtra = ds_size - dl_size
r_xtra = ds_size - dr_size
indice_L = random.sample(range(main_size), l_xtra)
indice_R = random.sample(range(main_size), r_xtra)

for i in indice_L:
  if steering_recover[i] < -0.15:
    d_left.append(right[i])
    a_left.append(steering_recover[i] - 0.27)

for i in indice_R:
  if steering_recover[i] > 0.15:
    d_right.append(left[i])
    a_right.append(steering_recover[i] + 0.27)

X_train = d_straight + d_left + d_right
y_train = np.float32(a_straight + a_left + a_right)

# Changing brightness randomly
def changeBrightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand = random.uniform(0.3,1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img 

# Flipping image around vertical axis
def flip(image, angle):
  new_image = cv2.flip(image,1)
  new_angle = angle*(-1)
  return new_image, new_angle

# Cropping image
def crop_resize(image):
  cropped = cv2.resize(image[60:140,:], (64,64))
  return cropped

def dataGenerator(batch_size):
	batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
	batch_angle = np.zeros((batch_size,), dtype = np.float32)
	while True:
		data, angle = shuffle(X_train, y_train)
		for i in range(batch_size):
			choice = int(np.random.choice(len(data),1))
			batch_train[i] = crop_resize(changeBrightness(mpimg.imread(data[choice].strip())))
			batch_angle[i] = angle[choice]*(1+ np.random.uniform(-0.10,0.10))
			flip_coin = random.randint(0,1)
			if flip_coin == 1:
				batch_train[i], batch_angle[i] = flip(batch_train[i], batch_angle[i])

		yield batch_train, batch_angle

# Validation
def checkingValidation(data, angle, batch_size):
    batch_train = np.zeros((batch_size, 64, 64, 3), dtype = np.float32)
    batch_angle = np.zeros((batch_size,), dtype = np.float32)
    while True:
      data, angle = shuffle(data,angle)
      for i in range(batch_size):
        rand = int(np.random.choice(len(data),1))
        batch_train[i] = crop_resize(mpimg.imread(data[rand].strip()))
        batch_angle[i] = angle[rand]
      yield batch_train, batch_angle

def main(_):
	data_generator = dataGenerator(128)
	valid_generator = checkingValidation(X_valid, y_valid, 128)

	input_shape = (64,64,3)
	model = Sequential()
	model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
	model.add(Convolution2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Convolution2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
	model.add(Activation('relu'))
	model.add(Flatten())
	model.add(Dense(80, W_regularizer = l2(0.001)))
	model.add(Dropout(0.5))
	model.add(Dense(40, W_regularizer = l2(0.001)))
	model.add(Dropout(0.5))
	model.add(Dense(16, W_regularizer = l2(0.001)))
	model.add(Dropout(0.5))
	model.add(Dense(10, W_regularizer = l2(0.001)))
	model.add(Dense(1, W_regularizer = l2(0.001)))
	adam = Adam(lr = 0.0001)
	model.compile(optimizer= adam, loss='mse', metrics=['accuracy'])
	model.summary()
	model.fit_generator(data_generator, samples_per_epoch = math.ceil(len(X_train)), nb_epoch=25, validation_data = valid_generator, nb_val_samples = len(X_valid))
	try:
		plot_model(model, to_file=os.path.join('plot', 'model.png')) 
	except AssertionError as error:
		print('>>>> ', error)

	print('Done training the model.')

	model_json = model.to_json()
		with open("model.json", "w") as json_file:
		json_file.write(model_json)
		model.save_weights("model.h5")
		print("Saved model to disk")

if __name__ == '__main__':
	tf.app.run()