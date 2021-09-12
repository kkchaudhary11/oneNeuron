import numpy as np

class Perceptron:
  def __init__(self,eta,epochs):
    self.weights = np.random.randn(3) * 1e-4
    print(f"Initial weighs : {self.weights}\n")
    self.eta = eta
    self.epochs = epochs
  
  def activationFunction(self,inputs,weights):
    z = np.dot(inputs, weights)
    return np.where(z > 0, 1, 0)
  
  def fit(self, X, y):
    self.X = X
    self.y = y

    X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
    print(f"X with bias : {X_with_bias}\n")

    for epoch in range(self.epochs):
      print("--"*10)
      print(f"Epoch : {epoch}")
      print("--"*10)

      y_hat = self.activationFunction(X_with_bias,self.weights)
      print(f"Predicated values after forward pass : \n{y_hat}")
      self.error = self.y - y_hat
      print(f"Error : \n{self.error}")
      self.weights = self.weights + self.eta * np.dot(X_with_bias.T,self.error)
      print(f"Updated weights afte epoch: {self.weights}")
      print("###"*10)

  def predict(self, X):
     X_with_bias = np.c_[X, -np.ones((len(X), 1))]
     return self.activationFunction(X_with_bias, self.weights)
  
  def total_loss(self):
    total_loss = np.sum(self.error)
    print(f"Total loss:{total_loss} ")
    return total_loss