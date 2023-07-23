import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from tensorflow.keras.datasets import cifar10



(x_train,y_train),(x_test,y_test)=cifar10.load_data()

x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0

model=keras.Sequential(
    [
        keras.Input(shape=(32,32,3)),
        ##Conv2D= to process 2D data =Images
        layers.Conv2D(30,3,padding='valid',activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64,3,padding='valid',activation='relu',kernel_regularizer=regularizers.L2(0.01)),
        layers.MaxPooling2D(),
        layers.Conv2D(128,3,activation='relu',strides=2),
        layers.Flatten(),
        layers.Dense(64,activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10),
    ])

model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(),
        metrics=["accuracy"])

model.fit(x_train,y_train,epochs=10,batch_size=32,verbose=2)
model.evaluate(x_test,y_test,batch_size=32,verbose=2)
