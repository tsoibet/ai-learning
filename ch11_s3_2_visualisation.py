import torch
from torch import nn
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

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

model.load_state_dict(torch.load('./sample11_3.pth'))

test_data = datasets.MNIST(
  root = "data",
  train = False, 
  download = True, 
  transform = ToTensor()
)
batch_size = 100
test_dataloader = data.DataLoader(test_data, batch_size=batch_size)

preds = torch.zeros((0, 10))

model.eval()
with torch.no_grad():
  for X, t in test_dataloader: 
    pred = model(X)
    preds = torch.concat((preds, pred))

for i in range(10):
  c = 0
  for (image, actual), pred in zip(test_data, preds):
    index = pred.argmax()
    if index != i:
      continue
    # first 3 are correct results and last 3 are wrong ones
    if (c < 3 and i == actual) or (c >= 3 and i != actual):
      plt.subplot(6, 10, i + 1 + c * 10) 
      plt.xticks([])
      plt.yticks([])
      plt.title(f'{index} / {actual}', {'fontsize': 'small'})
      plt.imshow(image.permute(1, 2, 0), cmap='gray_r')
      c += 1
      if c > 5:
        break

plt.show()