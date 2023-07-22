import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
print(tf.__version__)

# Initialization of Tensors
x= tf.constant(4, shape=(1,1))
x=tf.ones((3,3))
x=tf.zeros((2,2))
x=tf.eye(3)#identity matrix
##Normal distribution
x=tf.random.normal((3,3),mean=0,stddev=1)
x=tf.range(start=1,limit=10,delta=2)
x=tf.cast(x,dtype=tf.float64)
print(x)

#Indexing
print(x[1:])
print(x[::2])
print(x[::-1])#Inverse order


