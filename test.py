##
from skimage import io
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision #0.12.0
from dataproc import HappyWhalesDataset
import os
import dataproc
import numpy as np
import pickle
import time
#from cnvrg import Experiment
from sklearn.metrics import classification_report
import tqdm
import argparse
from torch.utils.data import Dataset


# Arguments
parser = argparse.ArgumentParser(description='PyTorch CNN HappyWhales')
parser.add_argument('--mode', type=str, default='test', metavar='N',
                    help='mode of evaluation')
parser.add_argument('--fold', type=int, default=2, metavar='N',
                    help='Number of the folder want to make test')
parser.add_argument('--model', type=str, default='ResNet50_Adam_lr1e-4Decay6_wd2e-4_Fold1.pth',
                    help='model on wich you want to take the test, must not be the same folder of  args.fold ')
parser.add_argument('--CNN', type=str, default='Resnet50',
                    help='model on wich you want to take the test, must not be the same folder of  args.fold ')
parser.add_argument('--img', type=str, default='7a38de7c97d300.jpg',
                    help='Image of the fold selected ')
parser.add_argument('--gpuID', type=int, default=0,
                    help='file on which to save model weights')

args = parser.parse_args()


mean2 = [0.4124,0.4564,0.5065]
std2 = [0.2245,0.2230,0.2336]

my_transform_fold2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.4124,0.4564,0.5065], std=[0.2245,0.2230,0.2336])
])

mean1 = [0.4108,0.4552,0.5053]
std1 = [0.2241,0.2227,0.2335]


my_transform_fold1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean1, std1)
])



test_dataset = HappyWhalesDataset(csv_file= "fold"+str(args.fold)+".csv", root_dir=str(os.path.join("data","fold"+str(args.fold))),transform=my_transform_fold1 )
test_loader = DataLoader(test_dataset, batch_size= 1,shuffle=False)




if args.CNN == "Resnet50":
    model = torchvision.models.resnet50(pretrained=False)
    model.fc = nn.Linear(2048,26)
else:
    model = torchvision.models.efficientnet_b7(pretrained =False)
    model.classifier[1] = nn.Linear(2560,26)
    '''model.features[7] = nn.Identity()
    model.classifier[1] = nn.Linear(384,26)'''


pretrained_dict = torch.load(os.path.join(args.model))
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict,strict=False)



def Test(model,test_dataloader, gpuID):
   
    net = model.cuda(gpuID) # this will bring the network to GPU if DEVICE is cuda
    net.eval() # Set Network to evaluation mode

    dicc = pickle.load(open("dicc_clases.pkl","rb"))
    keys = list(dicc.keys())
    print(dicc[keys[0]])
    print(keys[0])
    predsfinal = []
    labelsfinal = []
    contador = 0
    largo = len(test_dataloader)
    for images, labels in test_dataloader:
        images = images.cuda(gpuID)
        labels = labels

  # Forward Pass
        outputs = net(images)

  # Get predictions
        _, preds = torch.max(outputs.data, 1)

        predsfinal.append(list(preds.cpu().numpy()))  
        labelsfinal.append(list(labels.numpy()))  

        contador += 1
        print("imagen: "+ str(contador)+"/"+str(largo))


    labelsfinalF = []
    predsfinalF = []

    for i in range(len(predsfinal)):
        for j in range(len(keys)):
            if str(dicc[keys[j]]) == str(labelsfinal[i][0]):
                labelsfinalF.append(str(keys[j]))
                print("entrol")
            if str(dicc[keys[j]]) == str(predsfinal[i][0]):
                predsfinalF.append(str(keys[j]))
                print("entrop")



    print(classification_report(labelsfinalF,predsfinalF))

class HappyWhalesDatasetdemo(Dataset):
    def __init__(self,csv_file,root_dir,image, transform= None):
        self.annotations =pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image = image
        self.transform = transform
    

    def __len__(self):

        return len(self.annotations) #51033
    def __getitem__(self, index):
        
        img_path = os.path.join(self.root_dir,self.image)
        image = io.imread(img_path)
        y_label = torch.tensor(int((self.annotations.loc[self.annotations['image'] == self.image])["species"]))

        if self.transform:
            image = self.transform(image)

        return(image,y_label)

if args.mode == "test":
    Test(model,test_loader,args.gpuID)

else:
    
    
    test_dataset = HappyWhalesDatasetdemo(csv_file= "fold"+str(args.fold)+".csv", root_dir=str(os.path.join("data","fold"+str(args.fold))),transform=my_transform_fold2,image= args.img)
    test_loader = DataLoader(test_dataset, batch_size= 1,shuffle=False)
    net = model.cuda(args.gpuID) # this will bring the network to GPU if DEVICE is cuda
    net.eval() # Set Network to evaluation mode

    dicc = pickle.load(open("dicc_clases.pkl","rb"))
    keys = list(dicc.keys())
    predsfinal = []
    labelsfinal = []
    for images, labels in test_loader:
        images = images.cuda(args.gpuID)
        labels = labels

        outputs = net(images)
        _, preds = torch.max(outputs.data, 1)

        predsfinal =preds.cpu().numpy()
        labelsfinal =labels.numpy()
        predsfinal = keys[int(predsfinal)]
        labelsfinal = keys[int(labelsfinal)]

        break

    print("La predicción es " +str(predsfinal) + " La anotación es "+ str(labelsfinal))
    if predsfinal == labelsfinal:
        print("La predicción fue correcta")
    else:
        print("La predicción No fue correcta")