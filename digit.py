import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2

def create_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    IMG_SIZE=28
    print(x_train)
    x_trainr = np.array(x_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    x_testr = np.array(x_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


    model = tf.keras.models.Sequential()

    #Convolution Layer 1
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape = x_trainr.shape[1:]))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #Convolution Layer 2
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #Convolution Layer 3
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # Fully Connected Layer 1
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation("relu"))

    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.Activation("relu"))

    # Fully Connected Layer 2
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Activation("relu"))

    # Last Fully Connected Layer
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Activation("softmax"))

    #model.summary()

    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    #Training
    model.fit(x_trainr, y_train, epochs=6, validation_split = 0.3, batch_size=25)

    model.save('handwritten.model')

def predictDigit(img):
    model = tf.keras.models.load_model('handwritten.model')

    img = np.invert(img)
    plt.imshow(img)
    plt.show()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28,28), interpolation= cv2.INTER_AREA)

    newing = tf.keras.utils.normalize(resized, axis=1)
    newing = np.array(newing).reshape(-1, 28, 28, 1)

    predicions = model.predict(newing)
    return np.argmax(predicions)
