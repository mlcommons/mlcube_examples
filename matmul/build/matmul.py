import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

shape = (int(sys.argv[1]), int(sys.argv[1]))

a = tf.random.normal(shape=shape)
b = tf.random.normal(shape=shape)

x = tf.matmul(b,a)

print(x) 

startTime = datetime.now()

print("\n" * 5)
print("Shape:", shape)
print("Time taken:", datetime.now() - startTime)
print("\n" * 5)
