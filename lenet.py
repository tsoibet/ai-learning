from torch import nn

class LeNet(nn.Module): 
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(1, 6, 5, 1, 2), # Convolution layer 1
      nn.ReLU(),
      nn.MaxPool2d(2, 2), # Pooling layer 1
      nn.Conv2d(6, 16, 5, 1), # Convolution layer 2
      nn.ReLU(),
      nn.MaxPool2d(2, 2), # Pooling layer 2
      nn.Flatten(),
      nn.Linear(16 * 5 * 5, 120), # Fully Connected Layer 1
      nn.ReLU(),
      nn.Linear(120, 84), # Fully Connected Layer 2
      nn.ReLU(),
      nn.Linear(84, 10) # Fully Connected Layer 3
    )
  def forward(self, x):
    return self.layers(x)

if __name__ == '__main__':
  from torchinfo import summary
  model = LeNet()
  print(model)
  summary(model, input_size=(100, 1, 28, 28)) 