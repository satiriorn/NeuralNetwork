"""
import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([1,2,3,4,5,6], dtype=float)
ys = np.array([1.0,1.5,2.0,2.5,3.0,3.5], dtype=float)
model.fit(xs, ys, epochs=1000)

print(model.predict([7.0]))

import tensorflow as tf
from os import path, getcwd, chdir

# DO NOT CHANGE THE LINE BELOW. If you are developing in a local
# environment, then grab mnist.npz from the Coursera Jupyter Notebook
# and place it inside a local folder and edit the path to that location
path = f"{getcwd()}/../tmp2/mnist.npz"
# GRADED FUNCTION: train_mnist
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
def train_mnist():

    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    training_images=training_images/255.0
    test_images=test_images/255.0
    callbacks = myCallback()
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
      tf.keras.layers.Flatten(input_shape = (28,28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        # YOUR CODE SHOULD END HERE
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])
    # model fitting
    return history.epoch, history.history['acc'][-1]
"""

import os, signal, keras.models, time
import zipfile
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

def addzip():
    local_zip = 'cats_and_dogs.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('/')
    zip_ref.close()
    base_dir = '/cats_and_dogs'
    return base_dir


def validationdir(base_dir):
    validation_dir = os.path.join(base_dir, 'validation')
    return validation_dir

def traindir(base_dir):
    train_dir = os.path.join(base_dir, 'train')
    return train_dir

def shape():
    """150 x 150
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 150x150 with 3 bytes color
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    """#300 x 300
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(512, activation='relu'),

        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    return model

def traning(model, train_dir, validation_dir):
    print(validation_dir)
    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics = ['accuracy'])

    train_datagen = ImageDataGenerator( rescale = 1.0/255. )
    test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=64,
                                                        class_mode='binary',
                                                        target_size=(300, 300))

    validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                             batch_size=64,
                                                             class_mode  = 'binary',
                                                             target_size = (300, 300))
    callbacks = myCallback()
    history = model.fit_generator(train_generator,
                                  epochs=10,
                                  verbose=1,
                                  validation_data=validation_generator,
                                  callbacks=[callbacks])
    model.save('training/modelCatDogv2.h5')
    return history

def load_model():
    model = keras.models.load_model('training/modelCatDogv2.h5')
    model.summary()
    return model

def create_model(train_dir, validation_dir):
    model = shape()
    traning(model, train_dir, validation_dir)
    return model

def Main():
    base_dir = addzip()
    validation_dir=validationdir(base_dir)
    train_dir = traindir(base_dir)
    model=create_model(train_dir,validation_dir)
    model.history()
    #model = load_model()
    while(True):
        p= 'photo.jpg'
        img = image.load_img(p, target_size=(300, 300))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        classes = model.predict(images, batch_size=10)

        print(classes[0])

        if classes[0] > 0:
            print(p + " is a dog")

        else:
            print(p + " is a cat")
        time.sleep(5)

def kill():
    os.kill(     os.getpid() ,
             signal.SIGTERM
           )
if __name__ == '__main__':
    Main()