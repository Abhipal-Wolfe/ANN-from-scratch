import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('dataset_NN.csv')

for column, item in data.iteritems():
    if column != 'Class':
        data[column] = (data[column]-data[column].min())/(data[column].max()-data[column].min())

print(data)

data = np.array(data)
m, n = data.shape

np.random.shuffle(data)

data_dev = data[1400:m-1].T
Y_dev = data_dev[n-1]
Y_dev = np.int_(Y_dev)
X_dev = data_dev[0:n-2]

data_train = data[0:1400].T
Y_train = data_train[n-1]
Y_train = np.int_(Y_train)
X_train = data_train[0:n-2]
_,m_train = X_train.shape

def init_params():
    W1 = np.random.rand(20, 5) 
    b1 = np.random.rand(20, 1) 
    W2 = np.random.rand(15, 20) 
    b2 = np.random.rand(15, 1)
    W3 = np.random.rand(11, 15)
    b3 = np.random.rand(11, 1)
    losses = []
    accuracies=[]
    return W1, b1, W2, b2, W3, b3, accuracies,losses

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = (1 / (m_train)) * dZ3.dot(A2.T)
    db3 = (1 / (m_train)) * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = (1 / (m_train)) * dZ2.dot(A1.T)
    db2 = (1 / (m_train)) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / (m_train)) * dZ1.dot(X.T)
    db1 = (1 / (m_train)) * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def get_loss(predictions,Y):
    return np.sum(abs(predictions-Y)) / Y.size
    

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3, accuracies, losses = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 1000 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A3)
            print(predictions, Y)
            print(get_accuracy(predictions, Y))
            print(get_loss(predictions, Y))
            accuracies.append(get_accuracy(predictions, Y))
            losses.append(get_loss(predictions, Y))
    return W1, b1, W2, b2, W3, b3, accuracies, losses

W1, b1, W2, b2, W3, b3, accuracies, losses = gradient_descent(X_train, Y_train, (10e-2)/2, 12000)


def test_accuracy_loss(X, Y, W1, b1, W2, b2, W3, b3):
    Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    print("Testing accuracy " + str(get_accuracy(predictions, Y)))
    print("Testing loss " + str(get_loss(predictions, Y)))
            


plt.style.use("ggplot")
plt.figure()
plt.plot(accuracies)
plt.title("Training accuracy")
plt.xlabel("Epoch (*100)")
plt.ylabel("Accuracy")
plt.show()

plt.plot(losses)
plt.title("Training Loss" )
plt.xlabel("Epoch (*100)")
plt.ylabel("Loss")
plt.show()


test_accuracy_loss(X_dev, Y_dev, W1, b1, W2, b2, W3, b3)
