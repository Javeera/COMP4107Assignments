# Name this file assignment2.py when you submit
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

  batches_per_epoch = x_train.shape[0] // batch_size
  
  model = Salary_Prediction()

  loss_fn = torch.nn.MSELoss()

  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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


if __name__ == "__main__":
  mlb_position_player_salary("baseball.txt")