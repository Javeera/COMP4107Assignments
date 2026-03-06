# Name this file assignment3.py when you submit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# PyTorch dataset for the UWaveGestureLibrary dataset
class UWaveGestureLibraryDataset(torch.utils.data.Dataset):

  def __init__(self, dataset_filepath):
    # dataset_filepath is the full path to a file containing data
    self.samples = []
    self.labels = []

    with open(dataset_filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split(":")

        x_vals = [float(v) for v in parts[0].split(",") if v != ""]
        y_vals = [float(v) for v in parts[1].split(",") if v != ""]
        z_vals = [float(v) for v in parts[2].split(",") if v != ""]

        label = int(float(parts[3]))

        # shape = (3, 315)
        data = torch.tensor([x_vals, y_vals, z_vals], dtype=torch.float32)

        # one-hot encode label
        y = torch.zeros(8)
        y[label - 1] = 1.0

        self.samples.append(data)
        self.labels.append(y)
    # Return nothing    

  def __len__(self):
    # num_samples is the total number of samples in the dataset
    return len(self.samples)


  def __getitem__(self, index):
    # index is the index of the sample to be retrieved
    
    # x is one sample of data
    x = self.samples[index]
    # y is the label associated with the sample
    y = self.labels[index]
    return x, y


# A function that creates a cnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_cnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data

  # model is a trained cnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
  return model, training_performance, validation_performance


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.GRU(
            input_size=3,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.fc1 = nn.Linear(128 * 2, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 8)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # (batch, 315, 3)

        output, h = self.rnn(x)

        # last forward + last backward hidden states
        h_forward = h[-2]
        h_backward = h[-1]
        h = torch.cat([h_forward, h_backward], dim=1)

        x = self.fc1(h)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# A function that creates an rnn model to predict which class a sequence corresponds to
def u_wave_gesture_library_rnn_model(training_data_filepath):
  # training_data_filepath is the full path to a file containing the training data
  # model is a trained rnn model to predict which class a sequence corresponds to
  # training_performance is the performance of the model on the training set
  # validation_performance is the performance of the model on the validation set
    #torch.manual_seed(0) #for testing

    dataset = UWaveGestureLibraryDataset(training_data_filepath)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    #generator = torch.Generator().manual_seed(0) #for testing
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        #generator=generator #for testing
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = RNNModel()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    epochs = 30

    for epoch in range(epochs):
        model.train()

        for x, y in train_loader:
            labels = torch.argmax(y, dim=1)

            preds = model(x)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    training_performance = evaluate(model, train_loader)
    validation_performance = evaluate(model, val_loader)

    return model, training_performance, validation_performance

#helper to measure the accuracy
def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for x, y in loader:

            labels = torch.argmax(y, dim=1)

            preds = model(x)

            predicted = torch.argmax(preds, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total

if __name__ == "__main__":

    model, train_perf, val_perf = u_wave_gesture_library_rnn_model(
        "UWaveGestureLibrary_TRAIN.csv"
    )

    print("Training accuracy:", train_perf)
    print("Validation accuracy:", val_perf)