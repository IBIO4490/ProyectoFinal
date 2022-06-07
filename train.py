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
import time
from sklearn.metrics import classification_report



##
model = torchvision.models.resnet50(pretrained=False)
print(model)
##
model.fc = nn.Linear(2048,26)
print(model)

pretrained_dict = torch.load(os.path.join("ResNet50_Adam_lr1e-4Decay5_wd2e-4_Fold1.pth"))
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)


##
#Load data

mean2 = [0.4124,0.4564,0.5065]
std2 = [0.2245,0.2230,0.2336]

my_transform_fold2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    #x
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.Normalize(mean = [0.4124,0.4564,0.5065], std=[0.2245,0.2230,0.2336])
])

mean1 = [0.4108,0.4552,0.5053]
std1 = [0.2241,0.2227,0.2335]


my_transform_fold1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    #transforms.Resize(256),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.Normalize(mean1, std1)
])

train_dataset = HappyWhalesDataset(csv_file= "fold1.csv", root_dir=str(os.path.join("data","fold1")),transform=my_transform_fold1)
train_loader = DataLoader(train_dataset, batch_size= 5,shuffle= True)

##

'''def mean_std1(dataset,batch):
    loader = DataLoader(dataset, batch_size=batch)
    num_pixels = len(dataset)*512*512

    total_sum_R = 0
    total_sum_G = 0
    total_sum_B = 0
    for batch in loader: total_sum_R+= batch[0][:,0,:,:].cuda(1).sum(); total_sum_G+= batch[0][:,1,:,:].cuda(1).sum();total_sum_B+= batch[0][:,2,:,:].cuda(1).sum()

    meanR = total_sum_R/num_pixels
    meanG = total_sum_G/num_pixels
    meanB = total_sum_B/num_pixels

    sum_square_error_R = 0
    sum_square_error_G = 0
    sum_square_error_B = 0
    
    for batch in loader: sum_square_error_R+= ((batch[0][:,0,:,:].cuda(1)-meanR).pow(2)).sum(); sum_square_error_G+= ((batch[0][:,1,:,:].cuda(1)-meanG).pow(2)).sum(); sum_square_error_B+= ((batch[0][:,2,:,:].cuda(1)-meanB).pow(2)).sum()
    stdR= torch.sqrt(sum_square_error_R/num_pixels)
    stdG= torch.sqrt(sum_square_error_G/num_pixels)
    stdB= torch.sqrt(sum_square_error_B/num_pixels)

    mean = (meanR,meanG,meanB)
    std = (stdR,stdG,stdB)

    return mean, std'''

#print(mean_std1(train_dataset,1000))
#print(mean_std1(test_dataset,1000))

##
def train(log_interval, model, gpuID, train_loader, epoch,lr, weightdecay, model_name, scheduler = True):
    model.train()
    model.cuda(gpuID)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay= weightdecay)
    #optimizer = torch.optim.ASGD(model.parameters(), lr, weight_decay= weightdecay)
    #optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay= weightdecay)
    criterion = torch.nn.CrossEntropyLoss()
    timeformat = '%Y-%m-%d %H:%M:%S'
    #e = Experiment()
    for epoc in range(epoch):


        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(gpuID), target.cuda(gpuID)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            timestr = time.strftime(timeformat, time.localtime())
            if (batch_idx+1) % log_interval == 0:
                print('{} Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(timestr,epoc,
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    1.   * batch_idx / len(train_loader), loss.item()))
                #e.log_metric("loss", Ys=[loss.item()])
            torch.save(model.state_dict(),str(model_name+".pth"))

lr = 1e-4
weightdecay = 0.0002
modelname= "ResNet50_Adam_lr1e-4Decay6_wd2e-4_Fold1"
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay= weightdecay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
train(10,model,0,train_loader,3,lr,weightdecay,modelname, scheduler)

