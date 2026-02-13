# Name this file assignment2.py when you submit
import numpy
import torch

# A function that implements a pytorch model following the provided description
class MultitaskNetwork(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # Code for constructor goes here
    self.fullyconnectedlayer1 = torch.nn.Linear(3, 5)
    self.fullyconnectedlayer2 = torch.nn.Linear(5, 4)
    self.head_a = torch.nn.Linear(4, 3)
    self.head_b = torch.nn.Linear(4, 3)
    self.relu = torch.nn.ReLU()
    self.softmax = torch.nn.Softmax(dim=1)

  def forward(self, x):
    # Code for forward method goes here
    h = self.relu(self.fullyconnectedlayer1(x))
    h = self.relu(self.fullyconnectedlayer2(h))
    out_a = self.softmax(self.head_a(h))
    out_b = self.softmax(self.head_b(h))
    return out_a, out_b


# A function that implements training following the provided description
def multitask_training(data_filepath):
  num_epochs = 100
  batch_size = 4

  data = numpy.loadtxt(data_filepath, delimiter=",")
  batches_per_epoch = int(data.shape[0] / batch_size)

  multitask_network = MultitaskNetwork()

  # Define loss function(s) here
  def categorical_cross_entropy_from_probs(y_true_one_hot, y_pred_probs, eps=1e-12):
    # y_true_one_hot: (B, 3) one-hot labels
    # y_pred_probs:   (B, 3) softmax probabilities
    y_pred_probs = torch.clamp(y_pred_probs, min=eps, max=1.0)
    ce = -(y_true_one_hot * torch.log(y_pred_probs)).sum(dim=1)  # sum over classes
    return ce.mean()

  # Define optimizer here
  base_lr = 0.05
  optimizer = torch.optim.SGD(multitask_network.parameters(), lr=base_lr)

  #cosine learning rate schedule across ALL batch updates
  total_steps = num_epochs * batches_per_epoch
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

  for epoch in range(num_epochs):
    #shuffle each epoch
    perm = numpy.random.permutation(data.shape[0])
    data_epoch = data[perm]

    for batch_index in range(batches_per_epoch):
      x = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 6:9], dtype=torch.float32)
      y_a = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 0:3], dtype=torch.float32)
      y_b = torch.as_tensor(data[batch_index * batch_size:(batch_index + 1) * batch_size, 3:6], dtype=torch.float32)

      y_pred_a, y_pred_b = multitask_network(x)

      # Compute loss here
      loss_a = categorical_cross_entropy_from_probs(y_a, y_pred_a)
      loss_b = categorical_cross_entropy_from_probs(y_b, y_pred_b)
      loss = loss_a + loss_b
      # Compute gradients here
      optimizer.zero_grad()
      loss.backward()
      # Update parameters according to SGD with learning rate schedule here
      optimizer.step()
      scheduler.step()

  # A trained torch.nn.Module object
  return multitask_network

#Question 3
# A function that creates a pytorch model to predict the salary of an MLB position player
def mlb_position_player_salary(filepath):
  # filepath is the path to an csv file containing the dataset
  # model is a trained pytorch model for predicting the salary of an MLB position player

  data = numpy.loadtxt(filepath, delimiter=",", skiprows=1).astype(numpy.float32) #load data, first row is headers so skip

  #last line is salary
  #convert to tensors
  y_np = data_np[:, 0:1]     # shape (N, 1)
  X_np = data_np[:, 1:]      # shape (N, 16)

  X = torch.as_tensor(X_np, dtype=torch.float32)
  y = torch.as_tensor(y_np, dtype=torch.float32)

  #split data 80 training 20 validation
  N = X.shape[0]
  perm = torch.randperm(N)
  X = X[perm]
  y = y[perm]

  n_train = int(0.80 * N)
  X_train = X[:n_train]
  y_train = y[:n_train]
  X_val = X[n_train:]
  y_val = y[n_train:]

  #tensor operations
  mu = X_train.mean(dim=0, keepdim=True)
  sigma = X_train.std(dim=0, keepdim=True)
  sigma = torch.where(sigma == 0, torch.ones_like(sigma), sigma)

  X_train = (X_train - mu) / sigma
  X_val = (X_val - mu) / sigma

  #define model (simple regression MLP)
  input_dim = X_train.shape[1]

  class SalaryRegressor(torch.nn.Module):
    def __init__(self, d_in):
      super().__init__()
      self.fc1 = torch.nn.Linear(d_in, 64)
      self.fc2 = torch.nn.Linear(64, 64)
      self.out = torch.nn.Linear(64, 1)
      self.relu = torch.nn.ReLU()

    def forward(self, x):
      x = self.relu(self.fc1(x))
      x = self.relu(self.fc2(x))
      return self.out(x)

  model = SalaryRegressor(input_dim)

  #loss + optimizer
  loss_fn = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  #training
   num_epochs = 300
  batch_size = 32

  for epoch in range(num_epochs):
    perm_train = torch.randperm(X_train.shape[0])
    X_train_shuf = X_train[perm_train]
    y_train_shuf = y_train[perm_train]

    for start in range(0, X_train_shuf.shape[0], batch_size):
      end = start + batch_size
      xb = X_train_shuf[start:end]
      yb = y_train_shuf[start:end]

      pred = model(xb)
      loss = loss_fn(pred, yb)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

  #validation
  model.eval()
  with torch.no_grad():
    val_pred = model(X_val)
    val_mse = loss_fn(val_pred, y_val)
    validation_performance = torch.sqrt(val_mse).item()


  # validation_performance is the performance of the model on a validation set
  return model, validation_performance
  