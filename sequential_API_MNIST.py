import os
#Ignoring information message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist


(x_train,y_train),(x_test,y_test)=mnist.load_data()
print(f'Shape of the input train data {x_train.shape}')
print(f'Shape of output train data {y_train.shape}')
print(f'shape of the test data {x_test.shape}')


## reshaping input

x_train=x_train.reshape(-1,28*28).astype("float32") / 255.0
x_test=x_test.reshape(-1,28*28).astype("float32") / 255.0
print(f'Shape of the input train data {x_train.shape}')

##Modeling
#########SEQUENTIAL API###########

model=keras.Sequential(
    [   keras.Input(shape=(28*28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10)


    ]
)


##Other Option
## the good thing about this option that you debug yusing print(model.summary()) after every layer especially in a complex model.
model=keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))

##Specify the model configuration
model.compile(
    ## we used the SparseCategoricalCrossEntropy because the labels are integers means they are not one hot-encoded
    ## from_logits=True
    loss=keras.losses.SparseCategoricalCrossentropy( from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

#print(model.summary())
##First layers params=28*28*512+512
##second layer params=512*256+256
## last layer params=256*10+10

import sys
sys.exit()
##fitting the model
##batch_size=32 ==> number of samples that will be propagated
##epochs=5 ==> number of times the entire dataset will be passed through the network suring training
##verbose=2 ==> Display one line per epochn showing the progress and metrics
model.fit(x_train,y_train,batch_size=32,epochs=5,verbose=2)

## model evaluation
model.evaluate(x_test,y_test,batch_size=32,verbose=2)




##SUMMARY ABOUT THE SEQUENTIAL API
#(+)convenient == appropiate why ?
#Simple/readable/quickly prototyped

#(-) not flexible
# single input-output path
# can't create layers interconnected in arbitrary way


#########FUNCTIONAL API###########
#handle models with non-linear topology, shared layers and even multiple inputs outputs.
