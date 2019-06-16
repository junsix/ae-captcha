from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
from keras.utils.np_utils import to_categorical
from keras.layers import Activation, Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose, BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import os.path
import glob

from imutils import paths
import cv2

from keras.preprocessing import image
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

np.random.seed(1337)

# Image folders.
CAPTCHA_IMAGE_FOLDER_TRAIN = "./images/char-1-epoch-2000/train"
CAPTCHA_IMAGE_FOLDER_TEST = "./images/char-2-epoch-10/test"

# List of captchas.
x_train = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER_TRAIN, "*"))
x_test = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER_TEST, "*"))

data = []
target = []


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


data = np.array(data, dtype="float") / 255.0
target = np.array(target, dtype="float") / 255.0


data_test = []
target_test = []

for image_file in paths.list_images(CAPTCHA_IMAGE_FOLDER_TEST):
    # Load the image.
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, dsize=(60, 100), interpolation=cv2.INTER_AREA)
    # Add a third channel dimension to the image.
    image = np.expand_dims(image, axis=2)

    # Get the folder name (ie. the true character value).
    label = image_file.split(os.path.sep)[-1][0]
    # Add the image and char to the dictionary.
    ori = image_file.split(os.path.sep)[-1].split('_')[-1]
    if ori == 'ori.png':
        target_test.append(image)
    else :
        data_test.append(image)

data_test = np.array(data_test, dtype="float") / 255.0
target_test = np.array(target_test, dtype="float") / 255.0

x_train = data
y_train = target
x_test = data_test
y_test = target_test

def train_val_split(x_train, y_train):
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[:int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]

x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
print(x_train.shape, x_val.shape)

class Autoencoder():
    def __init__(self):
        self.img_rows = 100
        self.img_cols = 60
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        optimizer = Adam(lr=0.001)
        
        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
        self.autoencoder_model.summary()
        SVG(model_to_dot(self.autoencoder_model, show_shapes=True).create(prog='dot', format='svg'))

    def build_model(self):
        input_layer = Input(shape=self.img_shape)
        
        # encoder
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        h = MaxPooling2D((2, 2), padding='same')(h)
        
        # decoder
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
        h = UpSampling2D((2, 2))(h)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)
        
        return Model(input_layer, output_layer)
    
    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20):
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.00001,
                                       patience=1,
                                       verbose=1, 
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[early_stopping])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    
    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds

ae = Autoencoder()
ae.train_model(x_train, y_train, x_val, y_val, epochs=20, batch_size=200)

preds = ae.eval_model(x_test)
preds_0 = preds[10] * 255.0
preds_0 = preds_0.reshape(100, 60)
x_test_0 = x_test[10] * 255.0
x_test_0 = x_test_0.reshape(100, 60)
plt.imshow(x_test_0, cmap='gray')
plt.imshow(preds_0, cmap='gray')
preds = ae.eval_model(x_test)
# Display the 1st 8 corrupted and denoised images
rows, cols = 1, 5
num = rows * cols

ori = y_test[:num] * 255.0
corrup = x_test[:num] * 255.0
denos = preds[:num] * 255.0

imgs = np.concatenate([ori, corrup, preds[:num]])
imgs = imgs.reshape((rows * 3, cols, 100, 60))
imgs = np.vstack(np.split(imgs, rows, axis=1))
imgs = imgs.reshape((rows * 3, -1, 100, 60))
imgs = np.vstack([np.hstack(i) for i in imgs])
imgs = (imgs * 255).astype(np.uint8)
plt.figure()
plt.axis('off')
plt.title('Original images: top rows, '
          'Corrupted Input: middle rows, '
          'Denoised Input:  third rows')
plt.imshow(imgs, interpolation='none', cmap='gray')
Image.fromarray(imgs).save('corrupted_and_denoised.png')
plt.show()