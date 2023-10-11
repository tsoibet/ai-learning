import torch

def train(dataloader, model, loss_fn, optimizer, step=100):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, t) in enumerate(dataloader):
    pred = model(X)
    loss = loss_fn(pred, t)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if batch % step == 0:
      loss, current = loss.item(), batch * len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset)
  num_of_batches = len(dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, t in dataloader:
      pred = model(X)
      t = t.squeeze_()
      test_loss += loss_fn(pred, t).item()
      correct += (pred.argmax(1) == t).sum().item()
  test_loss /= num_of_batches
  correct /= size
  accuracy = 100 * correct
  print(f"Test Error: \n Accuracy: {accuracy:>0.2f}%, Avg loss: {test_loss:>7f}n")
  return test_loss, accuracy