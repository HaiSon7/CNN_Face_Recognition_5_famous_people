import cv2
import numpy as np
import os
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models

LABELS = ['Bill Gates','Elon Musk','Jeff Bezos','Mark Zuckerberg','Steve Jobs']
dict = {"bill_gates":[1,0,0,0,0],"elon_musk":[0,1,0,0,0],"jeff_bezos":[0,0,1,0,0],
        "mark_zuckerberg":[0,0,0,1,0],"steve_jobs":[0,0,0,0,1]}

TRAIN_DATA = 'image/data/train'
TEST_DATA = 'image/data/valid'

x_train = []
y_train = []

x_test = []
y_test = []
def getData(dir_data,x,y):
    for item in os.listdir(dir_data):
        item_path = os.path.join(dir_data,item)
        for filename in os.listdir(item_path):
            filename_path = os.path.join(item_path,filename)
            x.append((cv2.imread(filename_path),dict[item]))
            y.append(item)
    return x,y

x_train,y_train = getData(TRAIN_DATA,x_train,y_train)
x_test,y_test = getData(TEST_DATA,x_test,y_test)

np.random.shuffle(x_train)
np.random.shuffle(x_train)
np.random.shuffle(x_train)

for i in range(len(x_train)):
	if x_train[i][0].shape != (70, 70, 3):
		print(x_train[i][0].shape)
		print(i)




#--------------------------------------------------------------------#

model_training_first = models.Sequential([
	layers.Conv2D(32,(3,3),input_shape = (70,70,3),activation = "relu"),
	layers.MaxPool2D((2,2)),
	layers.Dropout(0.15),

	layers.Conv2D(64,(3,3),activation = "relu"),
	layers.MaxPool2D((2,2)),
	layers.Dropout(0.2),

	layers.Conv2D(128,(3,3),activation = "relu"),
	layers.MaxPool2D((2,2)),
	layers.Dropout(0.2),

	layers.Flatten(),
	layers.Dense(1000,activation = 'relu'),
	layers.Dense(256,activation = 'relu'),
	layers.Dense(5,activation = 'softmax'),
])
#model_training_first.summary()

model_training_first.compile(optimizer="adam",
                             loss="categorical_crossentropy",
                             metrics=["accuracy"])

x_train_images = np.array([x[0] for _, x in enumerate(x_train)])
y_train_labels = np.array([x[1] for _, x in enumerate(x_train)])
model_training_first.fit(x_train_images, y_train_labels, epochs=10)

model_training_first.save("model_famous_people_h5")

