
import csv
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# read record from excel file

samples = []
with open('./data4/driving_log.csv') as csv_table:
    read_data = csv.reader(csv_table)
    for line in read_data:
        samples.append(line)

# read images from left, center and right camera
def get_feature_label(data_samples):
    image_path = './data4/IMG/'
    images = []
    measurements = []
    correction = 0.1  # correct steer angle for left and right camera
    
    for line in data_samples:
        # split the image name from the record 
        img_name_center = line[0].split('/')[-1]
        img_name_left = line[1].split('/')[-1]
        img_name_right = line[2].split('/')[-1]
        
        # read images with RGB and combine them to a list 
        img_center = mpimg.imread(image_path + img_name_center)
        img_left = mpimg.imread(image_path + img_name_left)
        img_right = mpimg.imread(image_path + img_name_right)
        image = [img_center, img_left, img_right]
        
        # read the steering angle for central camera
        steering_center = float(line[3])
        # modify steering angle for left camera
        steering_left = steering_center + correction
        # modify steering angle for right camera
        steering_right = steering_center - correction
        # combine them to a list
        steering = [steering_center, steering_left, steering_right]
        
        # append the images and steering angles to list
        measurements.extend(steering)
        images.extend(image)
    return images, measurements


from sklearn.model_selection import train_test_split

# 80% are training samples, 20% are valid samples
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# get training sample
images, measurements = get_feature_label(train_samples)

# data augumentation to training samples
aug_images = []
aug_measurements = []
for image, measurement in zip(images,measurements):
    aug_images.append(image)
    aug_measurements.append(measurement)
    # augument with mirror flection
    aug_images.append(cv2.flip(image,1)) 
    aug_measurements.append(measurement*(-1.0)) 

# training data features and labels after augumentation
train_features = np.array(aug_images)
train_labels = np.array(aug_measurements)

# valid data features and labels
images, measurements = get_feature_label(validation_samples)
validation_features = np.array(images)
validation_labels = np.array(measurements)

# set the default batch_size
BATCH_SIZE = 64

# create a Generator
def generator (sample_features, sample_labels, batch_size = 32):
    num_examples = len(sample_features)
    while True:
        for offset in range(0,num_examples,batch_size):
            batch_X_train = sample_features[offset:offset+batch_size]
            batch_y_train = sample_labels[offset:offset+batch_size]
            yield [batch_X_train, batch_y_train]

train_generator = generator (train_features, train_labels, batch_size = BATCH_SIZE)
validation_generator = generator (validation_features, validation_labels, batch_size = BATCH_SIZE)


from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Activation, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

### create the neural net work architecture
model = Sequential()
# normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# cropping the images 
model.add(Cropping2D(cropping=((70,25),(0,0))))
# first convolutional layer with max pooling and 'relu' activation
model.add(Convolution2D(32,3,3,input_shape = (160,320,3)))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))

# flatten the convolutional layer
model.add(Flatten())  

# use dropout 
model.add(Dropout(0.8))

# output layer
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',metrics = ['accuracy'])
history_object = model.fit_generator(train_generator, steps_per_epoch= int(np.ceil(len(train_samples)/BATCH_SIZE)), \
        validation_data=validation_generator, validation_steps=int(np.ceil(len(validation_samples)/BATCH_SIZE)), \
        epochs=4, verbose = 1)
model.save('model.h5')


### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

