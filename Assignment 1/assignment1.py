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


def main():
  # Constants from Q4
  x1, x2 = 3, -2
  y1, y2 = 0.5, -0.75

  # ---------- Q4(a) ----------
  # f(a,b) = 1/2 * ( (a*x1 + b - y1)^2 + (a*x2 + b - y2)^2 )
  def f_a(v):
    a, b = v
    return 0.5 * ((a*x1 + b - y1)**2 + (a*x2 + b - y2)**2)

  # gradient:
  # df/da = (a*x1 + b - y1)*x1 + (a*x2 + b - y2)*x2
  # df/db = (a*x1 + b - y1)    + (a*x2 + b - y2)
  def df_a(v):
    a, b = v
    r1 = a*x1 + b - y1
    r2 = a*x2 + b - y2
    da = r1*x1 + r2*x2
    db = r1 + r2
    return [da, db]

  x0_a = [0.0, 0.0]
  alpha_a = 0.1

  argmin_a, minval_a = gradient_descent(f_a, df_a, x0_a, alpha_a)
  print("Q4(a)")
  print("  initial guess:", x0_a)
  print("  learning rate:", alpha_a)
  print("  (a, b) found: ", argmin_a)
  print("  min f value: ", minval_a)

  # ---------- Q4(b) ----------
  # f(a,b) = 1/2 * ( (SiLU(a*x1 + b) - y1)^2 + (SiLU(a*x2 + b) - y2)^2 )

  def silu(z):
    return z / (1 + math.exp(-z))

  # SiLU'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
  def silu_prime(z):
    s = 1 / (1 + math.exp(-z))  # sigmoid(z)
    return s * (1 + z * (1 - s))

  def f_b(v):
    a, b = v
    z1 = a*x1 + b
    z2 = a*x2 + b
    return 0.5 * ((silu(z1) - y1)**2 + (silu(z2) - y2)**2)

  # gradient:
  # df/da = (SiLU(z1)-y1)*SiLU'(z1)*x1 + (SiLU(z2)-y2)*SiLU'(z2)*x2
  # df/db = (SiLU(z1)-y1)*SiLU'(z1)    + (SiLU(z2)-y2)*SiLU'(z2)
  def df_b(v):
    a, b = v
    z1 = a*x1 + b
    z2 = a*x2 + b

    r1 = silu(z1) - y1
    r2 = silu(z2) - y2

    d1 = silu_prime(z1)
    d2 = silu_prime(z2)

    da = r1*d1*x1 + r2*d2*x2
    db = r1*d1 + r2*d2
    return [da, db]

  x0_b = [0.0, 0.0]
  alpha_b = 0.01

  argmin_b, minval_b = gradient_descent(f_b, df_b, x0_b, alpha_b)
  print("\nQ4(b)")
  print("  initial guess:", x0_b)
  print("  learning rate:", alpha_b)
  print("  (a, b) found: ", argmin_b)
  print("  min f value: ", minval_b)

main()