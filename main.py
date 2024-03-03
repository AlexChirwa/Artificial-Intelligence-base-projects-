import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.python.keras import activations
from tensorflow.python.keras.activations import softmax
data=tf.keras.datasets.mnist
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.layers import Dense

(x_train,y_train),(x_test,y_test)=data.load_data()

x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=5)

loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

for x in range(1,5):
    img=cv.imread(f'C:/Users/Alex Joshua Chirwa/PycharmProjects/Recognize Handwritten Digit/Data/{x}.png', cv.IMREAD_GRAYSCALE)
    img=cv.resize(img,(28,28))
    img=np.invert(np.array([img]))
    img=img/255.0
    prediction=model.predict(img)
    print("----------------")
    print("the predicted output is: ",np.argmax(prediction))
    print("----------------")
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()