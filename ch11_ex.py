import torch
torch.manual_seed(12345678)

import pandas as pd
from torch import nn
from torch import optim
from torch.utils import data
from nn_common import train, test

df = pd.read_csv('./data/sample11_5_1.csv')
train_df = df[:1000]
test_df = df[1000:]
train_x = torch.tensor(train_df[['x1', 'x2']].values, dtype=torch.float32)
train_t = torch.tensor(train_df['t'].values)
test_x = torch.tensor(test_df[['x1', 'x2']].values, dtype=torch.float32)
test_t = torch.tensor(test_df[['t']].values)
train_dataset = data.TensorDataset(train_x, train_t)
test_dataset = data.TensorDataset(test_x, test_t)
batch_size = 100
train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size)

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 80),
      nn.ReLU(),
      nn.Linear(80, 2)
    )
  def forward(self, x):
    return self.layers(x)

model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
losses = []
accuracies = []
epochs = 100
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(train_dataloader, model, loss_fn, optimizer, step=5)
  loss, acc = test(test_dataloader, model, loss_fn)
  losses.append(loss)
  accuracies.append(acc)
torch.save({'losses': losses, 'accuracies': accuracies}, 'sample11_ex.hst')
print("Done!")

#Visualisation of learning process
import matplotlib.pyplot as plt
history = torch.load('sample11_5_1.hst')
losses = history['losses']
accuracies = history['accuracies']
plt.subplot(2, 1, 1)
plt.title('Accuracies')
plt.plot(accuracies)
plt.subplot(2, 1, 2)
plt.title('Losses')
plt.plot(losses)
plt.show()