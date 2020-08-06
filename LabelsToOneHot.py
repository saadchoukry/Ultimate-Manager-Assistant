import numpy as np
import tensorflow as tf

class LabelsToOneHot:
    def __init__(self,npLabelsFilePath,classesNumber):
        self.Y = np.load(npLabelsFilePath)
        self.classesNumber = classesNumber
    
    def toOneHot(self):
        with tf.compat.v1.Session() as sess:
            C = tf.constant(self.classesNumber)
            one_hot_matrix = tf.one_hot(self.Y, self.classesNumber, axis=1)
            self.Y = sess.run(one_hot_matrix)
        return self
    
