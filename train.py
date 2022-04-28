##
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from dataproc import HappyWhalesDataset
import os
import dataproc
import numpy as np
import pickle
from cnvrg import Experiment


##
model = torchvision.models.efficientnet_b7(pretrained=True)
model.classifier[1] = nn.Linear(2560,30)
##
#Load data
my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((500,500)),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ToTensor(),
    #transforms.Normalize(mean = [], std=[])
])

train_dataset = HappyWhalesDataset(csv_file= "fold1.csv", root_dir=str(os.path.join("data","fold1")),transform=my_transform )
test_dataset = HappyWhalesDataset(csv_file= "fold2.csv", root_dir=str(os.path.join("data","fold2")),transform=my_transform )

train_loader = DataLoader(train_dataset, batch_size= 10, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size= 10, shuffle = True)

##
firstdata = len(train_dataset)

##
def train(log_interval, model, device, train_loader, epoch,lr, weightdecay):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay= weightdecay)
    criterion = torch.nn.CrossEntropyLoss()
    e = Experiment()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        #loss = F.nll_loss(output, target)
        #loss.backward()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                1.   * batch_idx / len(train_loader), loss.item()))
            e.log_metric("loss", Ys=[loss.item()])


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        1.   * correct / len(test_loader.dataset)))

# def Train(model,dataloaderT,dataloaderTe, epochs, lr, action, name,GPU):
#
#     device = torch.device("cuda:"+str(GPU) if torch.cuda.is_available() else "cpu")
#     if action == "save":
#         model.to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr)
#         criterion = torch.nn.CrossEntropyLoss()
#         for epoch in range(1, epochs+1):
#             model.train()
#             train_loss, train_acc = [], []
#             bar = dataloaderT
#             for X, y in bar:
#                 X, y = X.to(device), y.to(device)
#                 optimizer.zero_grad()
#                 y_hat = model(X)
#                 loss = criterion(y_hat, y)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss.append(loss.item())
#                 acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
#                 train_acc.append(acc)
#                 print(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")
#             bar = dataloaderTe
#             val_loss, val_acc = [], []
#             model.eval()
#             with torch.no_grad():
#                 for batch in bar:
#                     X, y = batch
#                     X, y = X.to(device), y.to(device)
#                     y_hat = model(X)
#                     loss = criterion(y_hat, y)
#                     val_loss.append(loss.item())
#                     acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
#                     val_acc.append(acc)
#                     #bar.set_description(f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}")
#             print(f"Epoch {epoch}/{epochs} loss {np.mean(train_loss):.5f} test_loss {np.mean(val_loss):.5f} acc {np.mean(train_acc):.5f} test_acc {np.mean(val_acc):.5f}")
#
#         pikle_out = open(name + ".pth", "wb")
#         pickle.dump(model, pikle_out)
#         pikle_out.close()
#     elif action == "load":
#         model = pickle.load(open(str(name) + ".pth", 'rb'))
#     return model