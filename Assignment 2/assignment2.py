# COMP 4107
# Javeera Faizi 101191910
# Julie Wechsler 101240968
# Name this file assignment2.py when you submit
from math import perm
import numpy
import torch

# A function that implements a pytorch model following the provided description
class MultitaskNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # Code for constructor goes here
    self.hidden_layer1 = torch.nn.Linear(3, 5)
    self.activation1 = torch.nn.ReLU()
    self.hidden_layer2 = torch.nn.Linear(5, 4)
    self.activation2 = torch.nn.ReLU()

    self.split_layer1 = torch.nn.Linear(4, 3)
    self.split_layer2 = torch.nn.Linear(4, 3)

    self.softmax = torch.nn.Softmax(dim=-1)

  def forward(self, x):
    # Code for forward method goes here
    x = self.hidden_layer1(x)
    x = self.activation1(x)
    x = self.hidden_layer2(x)
    x = self.activation2(x)

    #branch out for the output layer
    branch_1 = self.split_layer1(x)
    output_1 = self.softmax(branch_1)

    branch_2 = self.split_layer2(x)
    output_2 = self.softmax(branch_2)

    return output_1, output_2


# A function that implements training following the provided description
def multitask_training(data_filepath):
  num_epochs = 100
  batch_size = 4

  data = numpy.loadtxt(data_filepath, delimiter=",")
  batches_per_epoch = int(data.shape[0] / batch_size)

  multitask_network = MultitaskNetwork()

  # Define loss function(s) here
  def cross_entropy(true_y_1, predicted_y_1, true_y_2, predicted_y_2):

    loss_task_1 = -(true_y_1 * torch.log(predicted_y_1)).sum()
    loss_task_2 = -(true_y_2 * torch.log(predicted_y_2)).sum()
    total_loss = loss_task_1 + loss_task_2
    return total_loss
  
  # Define optimizer here
  optimizer = torch.optim.SGD(multitask_network.parameters(), lr=0.01)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

  
  for epoch in range(num_epochs):
    for batch_index in range(batches_per_epoch):
      x = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 6:9], dtype=torch.float32)
      y_a = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 0:3], dtype=torch.float32)
      y_b = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 3:6], dtype=torch.float32)

      y_pred_a, y_pred_b = multitask_network(x)

      # Compute loss here
      loss = cross_entropy(y_a, y_pred_a, y_b, y_pred_b)

      # Compute gradients here
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Update parameters according to SGD with learning rate schedule here
    scheduler.step()

  # A trained torch.nn.Module object
  return multitask_network




class Salary_Prediction(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.hidden_layer1 = torch.nn.Linear(16, 64)
    self.activation1 = torch.nn.ReLU()
    self.hidden_layer2 = torch.nn.Linear(64, 64)
    self.activation2 = torch.nn.ReLU()
    self.output = torch.nn.Linear(64, 1)


  def forward(self, x):
    x = self.hidden_layer1(x)
    x = self.activation1(x)
    x = self.hidden_layer2(x)
    x = self.activation2(x)
    x = self.output(x)

    return x


def mlb_position_player_salary(filepath):
  num_epochs = 300
  batch_size = 32

  data = numpy.loadtxt(filepath, delimiter=",", skiprows=1)

  #perm = numpy.random.permutation(data.shape[0])
  #data = data[perm]

  train_size = int(0.8 * data.shape[0])
  train_set = data[:train_size]
  test_set  = data[train_size:]

  x_train = train_set[:, 1:17]  
  y_train = train_set[:, 0:1] 

  x_test  = test_set[:, 1:17]
  y_test  = test_set[:, 0:1]

  x_train = torch.tensor(x_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.float32)
  x_test = torch.tensor(x_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.float32)

  x_train_mean = x_train.mean(dim=0)
  x_train_std = x_train.std(dim=0) + 1e-6

  x_train = (x_train - x_train_mean) / x_train_std
  x_test = (x_test - x_train_mean) / x_train_std

  #batches_per_epoch = x_train.shape[0] // batch_size
  batches_per_epoch = (x_train.shape[0] + batch_size - 1) // batch_size

  model = Salary_Prediction()
  loss_fn = torch.nn.MSELoss()

  model = Salary_Prediction()

  loss_fn = torch.nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


  for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_index in range(batches_per_epoch):

      start = batch_index * batch_size
      end = (batch_index + 1) * batch_size
      x = torch.as_tensor(x_train[start:end], dtype=torch.float32)
      y_true = torch.as_tensor(y_train[start:end], dtype=torch.float32)

      y_pred = model(x)

      # Compute loss here
      loss = loss_fn(y_pred, y_true)

      epoch_loss += loss.item()

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    scheduler.step()
    print("on epoch ", epoch)
    print("avg loss", epoch_loss / batches_per_epoch)


  model.eval() 
  with torch.no_grad():
    
    y_pred = model(x_test)
    loss_function = torch.nn.MSELoss()
    loss = loss_function(y_pred, y_test)

  print("TMSE:", loss.item())
  rmse = torch.sqrt(loss).item()
  print("RMSE:", rmse)

##############################################3

import matplotlib.pyplot as plt
import numpy
import torch

def _activation(name):
  name = name.lower()
  if name == "relu": return torch.nn.ReLU()
  if name == "tanh": return torch.nn.Tanh()
  if name == "sigmoid": return torch.nn.Sigmoid()
  if name == "leaky_relu": return torch.nn.LeakyReLU()
  if name == "elu": return torch.nn.ELU()
  if name == "gelu": return torch.nn.GELU()
  raise ValueError("Unknown activation: " + name)

def _build_regressor(input_dim, num_hidden_layers, neurons, activation_name):
  layers = []
  act = _activation(activation_name)

  d = input_dim
  for _ in range(num_hidden_layers):
    layers.append(torch.nn.Linear(d, neurons))
    layers.append(act)
    d = neurons

  layers.append(torch.nn.Linear(d, 1))  # regression output
  return torch.nn.Sequential(*layers)

def _rmse(model, X, y):
  model.eval()
  with torch.no_grad():
    pred = model(X)
    mse = torch.mean((pred - y) ** 2)
    return torch.sqrt(mse).item()

def run_salary_experiment(filepath,
                          neurons=64,
                          num_hidden_layers=2,
                          activation="relu",
                          epochs=300,
                          batch_size=32,
                          lr=1e-3,
                          seed=0):
  # Load baseball.txt (has header)
  data = numpy.loadtxt(filepath, delimiter=",", skiprows=1).astype(numpy.float32)
  y = torch.as_tensor(data[:, 0:1], dtype=torch.float32)
  X = torch.as_tensor(data[:, 1:], dtype=torch.float32)

  # Fixed shuffle + split: 70/15/15 (train/val/test)
  torch.manual_seed(seed)
  N = X.shape[0]
  perm = torch.randperm(N)
  X = X[perm]
  y = y[perm]

  n_train = int(0.70 * N)
  n_val   = int(0.15 * N)

  X_train = X[:n_train]
  y_train = y[:n_train]
  X_val   = X[n_train:n_train + n_val]
  y_val   = y[n_train:n_train + n_val]
  X_test  = X[n_train + n_val:]
  y_test  = y[n_train + n_val:]

  # Normalize using TRAIN stats only
  mu = X_train.mean(dim=0, keepdim=True)
  sigma = X_train.std(dim=0, keepdim=True)
  sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)

  X_train = (X_train - mu) / sigma
  X_val   = (X_val - mu) / sigma
  X_test  = (X_test - mu) / sigma

  # Model
  model = _build_regressor(X_train.shape[1], num_hidden_layers, neurons, activation)

  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # Train
  for _ in range(epochs):
    perm2 = torch.randperm(X_train.shape[0])
    Xs = X_train[perm2]
    ys = y_train[perm2]

    for start in range(0, Xs.shape[0], batch_size):
      end = min(start + batch_size, Xs.shape[0])
      xb = Xs[start:end]
      yb = ys[start:end]

      pred = model(xb)
      loss = loss_fn(pred, yb)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  # Metrics
  train_rmse = _rmse(model, X_train, y_train)
  val_rmse   = _rmse(model, X_val, y_val)
  test_rmse  = _rmse(model, X_test, y_test)

  return model, train_rmse, val_rmse, test_rmse



if __name__ == "__main__":
  mlb_position_player_salary("baseball.txt")

