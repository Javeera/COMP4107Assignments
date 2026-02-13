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

  data = numpy.loadtxt(filepath, delimiter=",") #load data

  #last line is salary
  X = data[:, :-1].astype(numpy.float32)
  y = data[:, -1].astype(numpy.float32).reshape(-1, 1)


  # validation_performance is the performance of the model on a validation set
  return model, validation_performance
  