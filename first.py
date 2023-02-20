# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 04:33:11 2023

@author: Ivan
"""

! pip install -q kaggle

! mkdir ~/.kaggle
! cp /content/kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets list

! kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

! unzip /content/gtsrb-german-traffic-sign.zip

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout

imgs_path = "/content/Train"

data_list = []
labels_list = []
classes_list = 43
for i in range(classes_list):
    i_path = os.path.join(imgs_path, str(i)) #0-42
    for img in os.listdir(i_path):
        im = Image.open(i_path +'/'+ img)
        im = im.resize((30,30))
        im = np.array(im)
        data_list.append(im)
        labels_list.append(i)
data = np.array(data_list)
labels = np.array(labels_list)
print("Done")

path = "/content/Train/0/00000_00000_00000.png"
img = Image.open(path)
img = img.resize((30, 30))
sr = np.array(img) 
plt.imshow(img)
plt.show()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(data, labels, test_size= 0.1)

print("training_shape: ", x_train.shape,y_train.shape)
print("testing_shape: ", x_test.shape,y_test.shape)

y_train = tf.one_hot(y_train,43)
y_test = tf.one_hot(y_test,43)

model = tf.keras.Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation="relu", input_shape= x_train.shape[1:]))
model.add((Conv2D(filters=32, kernel_size=(5,5), activation="relu")))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))
model.add((Conv2D(filters=64,kernel_size=(3,3),activation="relu"))) 
model.add((MaxPool2D(pool_size=(2,2))))
model.add(Dropout(rate=0.25))
model.add(Flatten()) 
model.add(Dense(256, activation="relu"))
model.add(Dropout(rate=0.40))
model.add(Dense(43, activation="softmax"))

model.compile(loss="categorical_crossentropy", 
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
history = model.fit(x_train,
                    y_train,
                    epochs=20,
                    batch_size=64,
                    validation_data=(x_test, y_test))

plt.figure(0)
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.title("Accuracy Graph")
plt.xlabel("epochs")
plt.ylabel("accuracy (0,1)")
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label="training loss")
plt.plot(history.history['val_loss'], label="val loss")
plt.title("Loss Graph")
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
test = pd.read_csv("/content/Test.csv")
test_labels = test['ClassId'].values
test_img_path = "/content"
test_imgs = test['Path'].values

test_data = []
test_labels = []

for img in test_imgs:
    im = Image.open(test_img_path + '/' + img)
    im = im.resize((30,30))
    im = np.array(im)
    test_data.append(im)

test_data = np.array(test_data)
print(test_data.shape)

import warnings
warnings.filterwarnings("ignore")
test_labels = test['ClassId'].values
test_labels

predict_x=model.predict(test_data)
classes_x=np.argmax(predict_x,axis=1)
accuracy=accuracy_score(test_labels,classes_x)

accuracy

classes_x

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#Fit the model

cf_matrix = confusion_matrix(test_labels, classes_x)
#print(cf_matrix)

import seaborn as sns
fig, ax = plt.subplots(figsize=(20,20))         # Sample figsize in inches
sns.heatmap(cf_matrix, annot=True, ax=ax)