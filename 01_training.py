import lightning as L
import torch
import torchvision
import numpy as np
from torch.nn import Sequential as S
from torch.nn import Conv2d as C
from torch.nn import Tanh as T
from torch.nn import AvgPool2d as A
from torch.nn import Linear as Li
from torch.nn import Flatten as F
import torchvision.transforms as transforms
import os

class LeNet5(torch.nn.Module):
  def __init__(self, n_channels=3, n_outputs=10):
    super().__init__()
    block = lambda ci,co,k=5: S(C(ci,co,k),T(),A(2,2))
    self.model = S(block(n_channels, 6),block(6, 16),S(C(16,120,5),T(),F(),Li(120,84),T(),Li(84,n_outputs)))
  def forward(self, x):
    return self.model(x)

class PyTorchComponent(L.LightningWork):
   def run(self):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    dataloaders = {
        'train': torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=10, pin_memory=True),
        'val': torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=10, pin_memory=True),
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet5().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        for phase in ['train', 'val']:
            if phase == 'train': model.train()
            else: model.eval()
            l,m=[],[]
            for x,y in dataloaders[phase]:
                x,y = x.to(device),y.to(device)
                model.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(x)
                    loss = criterion(output, y)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    l.append(loss.item())
                    m.append((output.argmax(1)==y).float().mean().item())
            print(f'epoch: {epoch}, phase: {phase}, loss: {np.mean(l):.5f} accuracy: {np.mean(m):.5f}')
    os.makedirs('outputs', exist_ok=True)
    torch.save(model.state_dict(), 'outputs/model.pt')


compute = L.CloudCompute('gpu', wait_timeout=60, spot=True)
componet = PyTorchComponent(cloud_compute=compute)
app = L.LightningApp(componet)