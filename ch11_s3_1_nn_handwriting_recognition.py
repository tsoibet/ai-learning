import torch
torch.manual_seed(12345678)

from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn
from torch import optim
from torch.utils import data
from nn_common import train, test

# download MNIST data if not exist
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

class NeuralNetwork(nn.Module):
  def __init__(self): 
    super().__init__()
    self.layers = nn.Sequential( 
      nn.Flatten(),
      nn.Linear(784, 10)
    )
  def forward(self, x): 
    return self.layers(x)
  
model = NeuralNetwork()

batch_size = 100
train_dataloader = data.DataLoader(train_data, batch_size=batch_size)
test_dataloader = data.DataLoader(test_data, batch_size=batch_size)

loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.Adam(model.parameters()) # better accuracy
losses = []
accuracies = []
epochs = 10
for t in range(epochs):
  # one pass through the entire training dataset
  print(f"Epoch {t+1}\n-------------------------------") 
  train(train_dataloader, model, loss_fn, optimizer) 
  loss, acc = test(test_dataloader, model, loss_fn)
  losses.append(loss)
  accuracies.append(acc)
torch.save(model.state_dict(), 'sample11_3.pth')
torch.save({'losses': losses, 'accuracies': accuracies}, 'sample11_3.hst')
print("Done!")

#Visualisation of learning process
import matplotlib.pyplot as plt
history = torch.load('./sample11_3.hst')
losses = history['losses']
accuracies = history['accuracies']
plt.subplot(2, 1, 1)
plt.title('Accuracies')
plt.plot(accuracies)
plt.subplot(2, 1, 2)
plt.title('Losses')
plt.plot(losses)
plt.show()