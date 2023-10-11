import torch
torch.manual_seed(12345678)

from torch import nn
from torch import optim
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor
from lenet import LeNet
from nn_common import train, test

train_data = datasets.MNIST(
  root = "data", 
  train = True, 
  download = True, 
  transform = ToTensor()
)
test_data = datasets.MNIST(
  root = "data",
  train = False, 
  download = True, 
  transform = ToTensor()
)
batch_size = 100
train_dataloader = data.DataLoader(train_data, batch_size=batch_size)
test_dataloader = data.DataLoader(test_data, batch_size=batch_size)

model = LeNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
losses = []
accuracies = []
epochs = 10
for t in range(epochs):
  print(f"Epoch {t+1}\n-------------------------------")
  train(train_dataloader, model, loss_fn, optimizer)
  loss, acc = test(test_dataloader, model, loss_fn)
  losses.append(loss)
  accuracies.append(acc)
torch.save(model.state_dict(), 'sample12_2_1.pth')
torch.save({'losses': losses, 'accuracies': accuracies}, 'sample12_2_1.hst')
print("Done!")

#Visualisation of learning process
import matplotlib.pyplot as plt
history = torch.load('./sample12_2_1.hst')
losses = history['losses']
accuracies = history['accuracies']
plt.subplot(2, 1, 1)
plt.title('Accuracies')
plt.plot(accuracies)
plt.subplot(2, 1, 2)
plt.title('Losses')
plt.plot(losses)
plt.show()