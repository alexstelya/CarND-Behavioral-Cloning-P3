import csv
from math import ceil
import os
import cv2
from enum import Enum
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Dense, Flatten, Conv2D, Lambda
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class ImagePosition(Enum):
    CENTER = 0
    LEFT = 1
    RIGHT = 2


def nVidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0)))),
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model


def get_images_paths(from_path, correction=0.25):
    def extract_path_with_measurement(position, correction):
        image_file = line[position.value].split('/')[-1].split('IMG\\')[-1]
        image_file_path = os.path.join(images_folder_path, image_file)
        if os.path.exists(image_file_path):
            images_paths.append(image_file_path)
            measurements.append(float(line[3]) + correction)
        return

    images_paths, measurements = [], []
    images_folders = [f for f in os.listdir(
        from_path) if os.path.isdir(os.path.join(from_path, f))]

    for folder in images_folders:
        driving_log = os.path.join(from_path, folder, "driving_log.csv")
        images_folder_path = os.path.dirname(driving_log)
        if os.path.isfile(driving_log):
            with open(driving_log) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    extract_path_with_measurement(ImagePosition.CENTER, 0)
                    extract_path_with_measurement(
                        ImagePosition.LEFT, correction)
                    extract_path_with_measurement(
                        ImagePosition.RIGHT, -correction)

    return images_paths, measurements


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while True:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for path, measurement in batch_samples:
                if os.path.exists(path):
                    image = cv2.imread(path)
                    images.append(image)
                    measurements.append(measurement)
#                     flipped_image = cv2.flip(image, 1)
#                     flipped_measurement = -measurement
#                     images.append(flipped_image)
#                     measurements.append(flipped_measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)


paths, measurements = get_images_paths("IMG")
batch_size = 32

samples = list(zip(paths, measurements))
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = nVidia_model()
model.compile(loss='mse', optimizer='adam')
model.summary()

history = model.fit_generator(train_generator,
                              steps_per_epoch=ceil(
                                  len(train_samples)/batch_size),
                              validation_data=validation_generator,
                              validation_steps=ceil(
                                  len(validation_samples)/batch_size),
                              epochs=3, verbose=1)

model.save('model.h5')
