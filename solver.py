import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
import datetime
import datasets.base as input_data
import os
import os.path
import glob
import matplotlib.pyplot as plt
from imutils import paths
import cv2

MAX_STEPS = 10000
BATCH_SIZE = 50

LOG_DIR = 'log/cnn1-run-%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

FLAGS = None


if __name__ == '__main__':
    # load data
    # Image folders.
    CAPTCHA_IMAGE_FOLDER_TRAIN = "./images/char-1-epoch-2000/train"
    CAPTCHA_IMAGE_FOLDER_TEST = "./images/char-1-epoch-2000/test"
    
    data = []
    target = []
    labels = np.array([])
    
    for image_file in paths.list_images(CAPTCHA_IMAGE_FOLDER_TRAIN):
        # Load the image.
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
        # Add a third channel dimension to the image.
        image = np.expand_dims(image, axis=2)
        
        # Get the folder name (ie. the true character value).
        label = image_file.split(os.path.sep)[-1][0]
        # Add the image and char to the dictionary.
        ori = image_file.split(os.path.sep)[-1].split('_')[-1]
        if ori == 'ori.png':
            target.append(image)
        else :
            data.append(image)
            labels = np.insert(labels, labels.size, label)
    
    
    data = np.array(data, dtype="float") / 255.0
    target = np.array(target, dtype="float") / 255.0
   
    LABEL_SIZE = labels.size
    IMAGE_HEIGHT = data.shape[1]
    IMAGE_WIDTH = data.shape[2]
    IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT
    print('label_size: %s, image_size: %s' % (LABEL_SIZE, IMAGE_SIZE))
    
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    batch_size = 128
    num_classes = 10
    epochs = 10

    def train_val_split(x_train, y_train):
        rnd = np.random.RandomState(seed=42)
        perm = rnd.permutation(len(x_train))
        train_idx = perm[:int(0.8 * len(x_train))]
        val_idx = perm[int(0.8 * len(x_train)):]
        return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]

    x_train, y_train, x_test, y_test = train_val_split(target, labels)
    print(x_train.shape, x_test.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    
    tb_hist = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=True)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1, 
                     validation_data=(x_test, y_test),
                     callbacks=[tb_hist])
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model acc')
    plt.ylabel('acc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    
    print('train finish')
    
    data = []
    target = []
    labels = np.array([])
    
    for image_file in paths.list_images(CAPTCHA_IMAGE_FOLDER_TEST):
        # Load the image.
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
        # Add a third channel dimension to the image.
        image = np.expand_dims(image, axis=2)
        
        # Get the folder name (ie. the true character value).
        label = image_file.split(os.path.sep)[-1][0]
        # Add the image and char to the dictionary.
        ori = image_file.split(os.path.sep)[-1].split('_')[-1]
        if ori == 'ori.png':
            target.append(image)
        else :
            data.append(image)
            labels = np.insert(labels, labels.size, label)
    
    
    data = np.array(data, dtype="float") / 255.0
    target = np.array(target, dtype="float") / 255.0
    
    x_test = target
    y_test = labels
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])