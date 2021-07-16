import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

a = tf.random.normal(shape=[10, 1])
b = tf.random.normal(shape=[10,10])

x = tf.matmul(b,a)

print(x) 

startTime = datetime.now()

print("\n" * 5)
print("Time taken:", datetime.now() - startTime)
print("\n" * 5)
