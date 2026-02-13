import numpy
import matplotlib.pyplot as plt
from assignment2 import run_salary_experiment

best = {"val": float("inf"), "neurons": None, "layers": None, "act": None, "epochs": None}

def update_best(val_rmse, neurons, layers, act, epochs):
  if val_rmse < best["val"]:
    best["val"] = val_rmse
    best["neurons"] = neurons
    best["layers"] = layers
    best["act"] = act
    best["epochs"] = epochs

################################################

def q4a_neurons(filepath):
  neuron_list = [8, 16, 32, 64, 128, 256]
  val_scores = []

  for n in neuron_list:
    _, _, val_rmse, _ = run_salary_experiment(filepath, neurons=n, num_hidden_layers=2, activation="relu", epochs=300, seed=0)
    val_scores.append(val_rmse)
    update_best(val_rmse, neurons=n, layers=2, act="relu", epochs=300)
    print("neurons =", n, "val RMSE =", val_rmse)

  plt.figure()
  plt.plot(neuron_list, val_scores, marker="o")
  plt.xlabel("Neurons per hidden layer")
  plt.ylabel("Validation RMSE")
  plt.title("Q4a: Validation RMSE vs Neurons")
  plt.savefig("q4a_neurons.png", dpi=200)
  plt.show()

  best_i = int(numpy.argmin(numpy.array(val_scores)))
  return neuron_list[best_i], val_scores[best_i]

def q4b_layers(filepath):
  layer_list = [1, 2, 3, 4, 5]
  val_scores = []

  for L in layer_list:
    _, _, val_rmse, _ = run_salary_experiment(filepath, neurons=64, num_hidden_layers=L, activation="relu", epochs=300, seed=0)
    val_scores.append(val_rmse)
    update_best(val_rmse, neurons=64, layers=L, act="relu", epochs=300)
    print("layers =", L, "val RMSE =", val_rmse)

  plt.figure()
  plt.plot(layer_list, val_scores, marker="o")
  plt.xlabel("Number of hidden layers")
  plt.ylabel("Validation RMSE")
  plt.title("Q4b: Validation RMSE vs Hidden Layers")
  plt.savefig("q4b_layers.png", dpi=200)
  plt.show()

  best_i = int(numpy.argmin(numpy.array(val_scores)))
  return layer_list[best_i], val_scores[best_i]

def q4c_epochs(filepath):
  epoch_list = [25, 50, 100, 200, 300, 500]
  val_scores = []

  for E in epoch_list:
    _, _, val_rmse, _ = run_salary_experiment(filepath, neurons=64, num_hidden_layers=2, activation="relu", epochs=E, seed=0)
    val_scores.append(val_rmse)
    update_best(val_rmse, neurons=64, layers=2, act="relu", epochs=E)
    print("epochs =", E, "val RMSE =", val_rmse)

  plt.figure()
  plt.plot(epoch_list, val_scores, marker="o")
  plt.xlabel("Epochs")
  plt.ylabel("Validation RMSE")
  plt.title("Q4c: Validation RMSE vs Epochs")
  plt.savefig("q4c_epochs.png", dpi=200)
  plt.show()

  best_i = int(numpy.argmin(numpy.array(val_scores)))
  return epoch_list[best_i], val_scores[best_i]

def q4d_activations(filepath):
  act_list = ["relu", "tanh", "sigmoid", "leaky_relu", "elu", "gelu"]
  val_scores = []

  for a in act_list:
    _, _, val_rmse, _ = run_salary_experiment(filepath, neurons=64, num_hidden_layers=2, activation=a, epochs=300, seed=0)
    val_scores.append(val_rmse)
    update_best(val_rmse, neurons=64, layers=2, act=a, epochs=300)
    print("activation =", a, "val RMSE =", val_rmse)

  plt.figure()
  plt.plot(act_list, val_scores, marker="o")
  plt.xlabel("Activation function")
  plt.ylabel("Validation RMSE")
  plt.title("Q4d: Validation RMSE vs Activation")
  plt.savefig("q4d_activations.png", dpi=200)
  plt.show()

  best_i = int(numpy.argmin(numpy.array(val_scores)))
  return act_list[best_i], val_scores[best_i]

def q4e_best(filepath, neurons, num_hidden_layers, activation, epochs):
  _, train_rmse, val_rmse, test_rmse = run_salary_experiment(
    filepath,
    neurons=neurons,
    num_hidden_layers=num_hidden_layers,
    activation=activation,
    epochs=epochs,
    seed=0
  )
  print("BEST SETTINGS:", neurons, "neurons,", num_hidden_layers, "layers,", activation, "activation,", epochs, "epochs")
  print("Train RMSE:", train_rmse)
  print("Val   RMSE:", val_rmse)
  print("Test  RMSE:", test_rmse)

###############################################

if __name__ == "__main__":
  fp = "baseball.txt"

  best_n, _ = q4a_neurons(fp)
  best_L, _ = q4b_layers(fp)
  best_E, _ = q4c_epochs(fp)
  best_A, _ = q4d_activations(fp)

  q4e_best(
    fp,
    neurons=best["neurons"],
    num_hidden_layers=best["layers"],
    activation=best["act"],
    epochs=best["epochs"]
  )