import pandas as pd
import os
import shutil
import random
import imageio
import files
import tensorflow as tf
# # data for +ve

# File_PATH = "metadata.csv"
# IMAGE_PATH = "images"

# df=pd.read_csv(File_PATH)
# c=0
# for (i,row) in df.iterrows():
#     if row['finding'] == "Pneumonia/Viral/COVID-19" and row['view'] =="PA" :
#         filename = row['filename']
#         image_path=os.path.join(IMAGE_PATH,filename)
#         image_copy_path= os.path.join('covid_target_images',filename)
#         shutil.copy(image_path,image_copy_path)
#         c+=1
# print(c)

# # data for -ve

# kaggle_path = "Normal"
# target = "non_covid_images"
# c=0
# image_name = os.listdir(kaggle_path)
# random.shuffle(image_name)

# for i in range(196):
#     name = image_name[i]
#     image_path = os.path.join(kaggle_path,name)
#     target_path = os.path.join(target,name)
#     shutil.copy2(image_path,target_path)
#     c=c+1
# print(c)




# TRAIN_PATH = "CovidDataset/Train"
# VAL_PATH = "CovidDataset/Test"

import numpy as np 
import matplotlib.pyplot as plt
import keras 
from keras.layers import *
from keras.models import *
from keras.preprocessing import image

# #cnn based model

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])

# model.summary()

#train

train_datagen = image.ImageDataGenerator(
    rescale= 1./255,
    shear_range= 0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_dataset = image.ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    'CovidDataset/Train',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

# print(train_generator.class_indices)

validation_generator = test_dataset.flow_from_directory(
    'CovidDataset/Val',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)


hist = model.fit_generator(
    train_generator,
    steps_per_epoch=8,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=2
)


tf.keras.models.save_model(model,"covid_detection.hdf5")

