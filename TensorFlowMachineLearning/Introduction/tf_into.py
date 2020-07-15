"""
There are two main components of tensorflow - graphs and sessions. Tensorflow creates a graph of computations. You create the partial
computuions, but it doesnt perform them. They are stored in a graph. These computations are stored in a graph, because they are linked
between each other. When we start a session, we start to execute the different aspects of the graph, starting from the base computations.
Somethimes we cannot access a variable, because it is still not evaluated.
"""
import tensorflow as tf
import numpy as np

"""
Tensor - generalization of vectors and matrices to potentially higher dimensions. Each tensor represents a partialy defined computation that
will eventually produce a value. TensorFlow programs work by building a graph of Tensor objects that details how tensors are related. Each tensor
has a data type and shape. Now we will show we create tensors :
"""
string = tf.Variable("Hello, my name is Boyan", tf.string)
number = tf.Variable(786, tf.int16)
floating_number = tf.Variable(8.91276, tf.float64)

"""
Tensors woth rank 0 are scalars. Now we will show how to create tensors with rank > 0 :
"""
rank1 = tf.Variable(["Hi"], tf.string)
rank2 = tf.Variable([["Hi", "Hello"], ["Guten tag", "Hey"]], tf.string)

"""
Changing shapes :
"""
tensor1 = tf.ones([1, 2, 3])
tensor2 = tf.reshape(tensor1, [2, 3, 1])
tensor3 = tf.reshape(tensor2, [3, -1]) # -1 tells the tensor to calculate the size of the dimension in that place

"""
Main types of tensors - Variables, Constants, Placeholders, SparseTensor. Except the Variable, all the others are immutable.
"""

"""
Creating a session and evaluating tensors :
"""
num1 = tf.Variable(1, tf.int16)
num2 = tf.Variable(2, tf.int16)
tensor = num1 + num2

print(tensor)
