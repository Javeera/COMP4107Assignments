# COMP 4107
# Javeera Faizi 101191910
# Julie Wechsler 101240968
import torch
import math

#QUESTION 1
# Write a function that simulates an artificial neuron. The neuron should use an aggregation function that is
# a weighted sum of inputs. The neuron should use the signmoid linear unit (SiLU) as the activation function.
# Do not include a bias term in the neuron.
#   SiLU(x) = x
#   1 + exp(−x)
# The function must be named “artificial_neuron”.
# The function should take two input arguments: 
#   (1) a list of inputs x of length n, and 
#   (2) a list of weights w of length n.
# The function should return one value: (1) the output of the neuron

# A function simulating an artificial neuron
def artificial_neuron(x, w):
  # x is a list of inputs of length n
  # w is a list of inputs of length n

  #weighted_sum = sum(xi * wi for xi, wi in zip(x, w))
  weighted_sum = sum(x[i] * w[i] for i in range(len(x)))
  output = weighted_sum / (1 + math.exp(-weighted_sum))# output is the output from the neuron
  return output


#QUESTION 2
# Write a function that performs standard gradient descent on a multi-variable function f, to estimate the
# minimum of the multi-variable function (note that your function might not find the global minimum). Your
# function may assume that the gradient ∇f is provided.
# The function must be named “gradient_descent”.
# The function should take four input arguments: 
#   (1) the multi-variable function f whose value may be computed by calling it with a list of coordinates of length n (i.e. a callback function), 
#   (2) ∇f, the gradient of f, whose value may be computed by calling it with a list of coordinates of length n (i.e. a callback function),
#   (3) a list of coordinates of length n, indicating an initial guess for the minimum on f, 
#   (4) the learning rate α.
# The function should return two values: 
#   (1) a list of coordinates of length n, indicating the minimum that was found, and 
#   (2) the value of f at the minimum that was found.

# A function performing gradient descent
def gradient_descent(f, df, x0, alpha):
  # f is a function that takes as input a list of length n
  # df is the gradient of f; it is a function that takes as input a list of length n
  # x0 is an initial guess for the input minimizing f
  # alpha is the learning rate

  x = x0.copy()
  max_iters = 1000
  for _ in range(max_iters):
    grad = df(x)
    x = [x[i] - alpha * grad[i] for i in range(len(x))]

  argmin_f = x
  min_f = f(x)

  # argmin_f is the input minimizing f
  # min_f is the value of f at its minimum
  return argmin_f, min_f


#QUESTION 3
# Write a function that creates a neural network in PyTorch by subclassing the "torch.nn.Module" class in
# PyTorch. The model must have at least one linear layer. The model may have any architecture, provided it
# has at least one linear layer.
# The function must be named “pytorch_module”.
# The function should take no input arguments.
# The function should return one value: (1) a PyTorch module object.
# For this question, you may use the PyTorch library and any other libraries it depends on.

# A function that returns a neural network module in PyTorch
def pytorch_module():
  class module(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
      return self.linear(x)
    # A pytorch module
  return module()