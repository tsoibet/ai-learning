from torch import nn
from torchinfo import summary

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(784, 1024),
      nn.ReLU(),
      nn.Linear(1024, 10)
    )
  def forward(self, x): 
    return self.layers(x)
  
model = NeuralNetwork()
summary(model, input_size=(100, 1, 784))