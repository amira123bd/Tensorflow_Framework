import os
#Ignoring information message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train,y_train),(x_test,y_test)=mnist.load_data()


x_train=x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test=x_test.reshape(-1,28*28).astype("float32") / 255.0

inputs=keras.Input(shape=(784))
x=layers.Dense(512, activation='relu',name='layer_1')(inputs)
x=layers.Dense(256, activation='relu', name='layer_2')(x)
outputs=layers.Dense(10, activation='softmax')(x)
model= keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)

model.evaluate(x_test,y_test,batch_size=32,verbose=2)

keras.utils.plot_model(model,"mnist_model.png")