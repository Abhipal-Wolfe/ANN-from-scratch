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
    W1 = np.random.rand(15, 5) 
    b1 = np.random.rand(15, 1) 
    W2 = np.random.rand(11, 15) 
    b2 = np.random.rand(11, 1)
    losses = []
    accuracies=[]
    return W1, b1, W2, b2,accuracies,losses


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / (m_train)) * dZ2.dot(A1.T)
    db2 = (1 / (m_train)) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1 / (m_train)) * dZ1.dot(X.T)
    db1 = (1 / (m_train)) * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def get_loss(predictions,Y):
    return np.sum(abs(predictions-Y)) / Y.size
    

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, accuracies, losses = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(predictions, Y)
            print(get_accuracy(predictions, Y))
            print(get_loss(predictions, Y))
            accuracies.append(get_accuracy(predictions, Y))
            losses.append(get_loss(predictions, Y))
    return W1, b1, W2, b2, accuracies, losses



def test_accuracy_loss(X, Y, W1, b1, W2, b2, iterations):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    print("Testing accuracy " + str(get_accuracy(predictions, Y)))
    print("Testing loss " + str(get_loss(predictions, Y)))
    
    

W1, b1, W2, b2, accuracies, losses = gradient_descent(X_train, Y_train, 0.10, 5000)


plt.style.use("ggplot")
plt.figure()
plt.plot(accuracies)
plt.title("Training accuracy")
plt.xlabel("Epoch (*50)")
plt.ylabel("Accuracy")
plt.show()

plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch (*50)")
plt.ylabel("Loss")
plt.show()

test_accuracy_loss(X_dev, Y_dev, W1, b1, W2, b2, 5000)




















