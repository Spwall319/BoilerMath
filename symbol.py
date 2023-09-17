import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
CATEGORIES = ["sub", "mul","div", "add"]
symbols = ["-", "*", "/", "+"]
def create_model_symbol():
    DATADIR = "C:\\Users\\spwal\\Downloads\\mathSymbols\\dataset"




    IMG_SIZE = 50


    training_data = []

    def create_training_data():
        for category in CATEGORIES:
            path = os.path.join(DATADIR, category)
            class_num = CATEGORIES.index(category)
            for img in os.listdir(path):
                try:
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except Exception as e:
                    pass

    create_training_data()

    import random

    random.shuffle(training_data)

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X = X/255.0

    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))

    model.add(Dense(4))
    model.add(Activation("sigmoid"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer ="adam", metrics=['accuracy'])

    y = np.array(y)

    model.fit(X, y, batch_size=30, epochs=6, validation_split = 0.3)

    model.save('handwrittenSymbol.model')

def predictSymbol(img):
    model = tf.keras.models.load_model('handwrittenSymbol.model')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    newy = cv2.resize(img, (50, 50))
    newy = newy.reshape(-1, 50, 50, 1)
    plt.imshow(img)
    plt.show()

    predicions = model.predict([newy])
    return symbols[np.argmax(predicions)]
