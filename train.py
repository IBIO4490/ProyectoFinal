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
from sklearn.metrics import classification_report

##
model = torchvision.models.resnet50(pretrained=True)
print(model)
##
model.fc = nn.Linear(2048,30)
print(model)
##
#Load data
my_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    #transforms.Normalize(mean = [], std=[])
])
'''transforms.RandomCrop((500,500)),
    transforms.RandomHorizontalFlip(p = 0.5),'''

train_dataset = HappyWhalesDataset(csv_file= "fold1.csv", root_dir=str(os.path.join("data","fold1")),transform=my_transform )
test_dataset = HappyWhalesDataset(csv_file= "fold2.csv", root_dir=str(os.path.join("data","fold2")),transform=my_transform )

train_loader = DataLoader(train_dataset, batch_size= 25541, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size= 25492, shuffle = True)
##
data_fold1 = next(iter(train_loader))
media_fold1, desviacion_fold1 = data_fold1[0].mean(),data_fold1[0].std()
print("Fold1:")
print(media_fold1, desviacion_fold1)

data_fold2 = next(iter(test_loader))
media_fold2, desviacion_fold2 = data_fold2[0].mean(),data_fold2[0].std()
print("Fold2:")
print(media_fold2, desviacion_fold2)


##
def train(log_interval, model, device, train_loader, epoch,lr, weightdecay, model_name):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay= weightdecay)
    criterion = torch.nn.CrossEntropyLoss()
    e = Experiment()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                1.   * batch_idx / len(train_loader), loss.item()))
            e.log_metric("loss", Ys=[loss.item()])
        torch.save(model.state_dict(),str(model_name+".pth"))


def Test(model,test_dataloader, gpuID):
    device = torch.device("cuda:"+gpuID if torch.cuda.is_available() else "cpu")
    net = model.to(device) # this will bring the network to GPU if DEVICE is cuda
    net.eval() # Set Network to evaluation mode

    dicc = pickle.load(open("dicc_clases.pkl","rb"))
    keys = list(dicc.keys())

    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

  # Forward Pass
        outputs = net(images)

  # Get predictions
        _, preds = torch.max(outputs.data, 1)

        preds = list(preds.numpy())
        labels = list(labels.numpy())

        for clase in keys:
            for j in range(len(preds)):
                if dicc[clase] == preds[j]:
                    preds[j] = str(clase)
                if dicc[clase] == labels[j]:
                    labels[j] = str(clase)



        print(classification_report(labels,preds))






# def test(model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#             correct += pred.eq(target.view_as(pred)).sum().item()
#
#     test_loss /= len(test_loader.dataset)
#
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         1.   * correct / len(test_loader.dataset)))


