import numpy as np
import tensorflow as tf
from tensorflow.Keras import layers, models

# Perceptron
def perceptron(x, w, b, activation):
    y= np.dot(x, w) + b
    return activation(y)
def step(z):
    return 1 if z >= 0 else 0
x = np.array([1, 0])
w = np.array([0.5, 0.5])
b = -0.2
print("Perceptron Output:", perceptron(x, w, b, step))

# Activation Functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def relu(x):
    return np.maximum(0, x)
def softmax(x):
    exps = np.exp(x - np.max(x))  
    return exps / np.sum(exps)

vals = np.array([-2, 0, 3])
print("\nActivation Functions:")
print("Sigmoid:", sigmoid(vals))
print("ReLU:", relu(vals))
print("Softmax:", softmax(vals))

# Backpropagation & Gradient Descent 
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data 
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights
np.random.seed(1)
w1 = np.random.rand(2, 2)   
w2 = np.random.rand(2, 1)  
lr = 0.5

# Training loop
for epoch in range(5000):
    hidden = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(hidden, w2))
    error = y - output
    d_output = error * sigmoid_derivative(output)
    d_hidden = d_output.dot(w2.T) * sigmoid_derivative(hidden)
    w2 += hidden.T.dot(d_output) * lr
    w1 += X.T.dot(d_hidden) * lr
print("\nFinal predictions (manual NN):\n", output)

# TensorFlow/Keras Example
model = models.Sequential([
    layers.Dense(2, input_dim=2, activation="sigmoid"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=500, verbose=0)
print("\nPredictions (TensorFlow/Keras):")
print(model.predict(X))
