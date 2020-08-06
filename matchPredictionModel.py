## Importing libraries
import numpy as np
from playersDfCleaner import PlayersDfCleaner as cleaner  
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

data = np.load("numpyMatches.npy")
data = data.astype('int64')
X = StandardScaler().fit_transform(data[:,:-1])
Y = data[:,-1] 

xTrain, xTest, yTrain, yTest = train_test_split(X, Y,test_size = 0.1, random_state = 0)

print("XTrain has shape: " + str(xTrain.shape))
print("XTest has shape: " + str(xTest.shape))


"""
SVM classifier
"""


## Training a linear SVM classifier 
from sklearn.svm import SVC 
print("Training.....")
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(xTrain, yTrain) 
svm_predictions = svm_model_linear.predict(xTest) 

## Model accuracy for xTest   
accuracy = svm_model_linear.score(xTest, yTest) 
print("Accuracy: "+ str(accuracy))


"""
DNN classifier
"""

"""
Building of a deep neural network
    Architecture : 3 hidden layers
        - Input : 3900 examples of 1368 features each
        - Hidden layer 1: 50 units (Linear -> Relu)
        - Hidden layer 2: 25 units (Linear -> Relu)
        - Output : 3 units (Linear -> Softmax)
"""


"""
import tensorflow as tf
import math
import matplotlib.pyplot as plt

## Defining functions in order to train a model in tensorflow2

    ## Creating placeholders for X and Y


def create_placeholders(n_x, n_y):
    X = tf.compat.v1.placeholder(dtype=tf.float32,shape=(n_x,None),name="X")
    Y = tf.compat.v1.placeholder(dtype=tf.float32,shape=(n_y,None),name="Y")
    return X, Y

    ## Initializing parameters: 
    # Xavier initialisation for W matrices and zeros for biases

def initialize_parameters():
    tf.random.set_seed(1234)                  
    W1 = tf.compat.v1.get_variable("W1", [200,1368], initializer = tf.initializers.GlorotUniform())
    b1 = tf.compat.v1.get_variable("b1", [200,1], initializer = tf.zeros_initializer())
    W2 = tf.compat.v1.get_variable("W2", [20, 200], initializer = tf.initializers.GlorotUniform())
    b2 = tf.compat.v1.get_variable("b2", [20, 1], initializer = tf.zeros_initializer())
    W3 = tf.compat.v1.get_variable("W3", [3, 20], initializer = tf.initializers.GlorotUniform())
    b3 = tf.compat.v1.get_variable("b3", [3, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

    ## Forward propagation
def forward_propagation(X, parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X),b1) 
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)                         
    Z3 = tf.add(tf.matmul(W3,A2),b3)                                               
    return Z3

    ## Computing cost : - Z3 is the output of forward propagation
    #                   - Y contains true labels
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    return cost


    ## Creating random mini-batches of size 64 (64 examples each)
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    return mini_batches

    ## Defining the model: gradient descent is used to optimize the cost function
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    tf.compat.v1.reset_default_graph()                # to be able to rerun the model without overwriting tf variables
    tf.compat.v1.set_random_seed(1)                   # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / minibatch_size
                
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
    
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        saver.save(sess, "bestPositionModel.ckpt")
        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
        C = tf.constant(3)
        one_hot_matrix = tf.one_hot(Y, C, axis=1)
        Y = sess.run(one_hot_matrix)

        
xTrain, xTest, yTrain, yTest = train_test_split(X, Y,test_size = 0.1, random_state = 0)

xTrain, xTest, yTrain, yTest = xTrain.T, xTest.T, yTrain.T, yTest.T
## Running training/test
model(xTrain, yTrain, xTest, yTest, learning_rate = 0.001,
          num_epochs = 500, minibatch_size = 128, print_cost = True)

"""

if __name__ == '__main__':
    pass
