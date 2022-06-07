import os
import gc
import cv2
import math
import copy
import time
import random
import argparse
# For data manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from torch.cuda.amp import autocast, GradScaler
import torchvision
from skimage import io
# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict
import torchvision.transforms as transforms



import warnings

warnings.filterwarnings("ignore")
# For descriptive error messages
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Arguments
parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--val_batch_size', type=int, default=6, metavar='N',
                    help='input batch size for testing (default: 6)')
parser.add_argument('--epochs', type=int, default=25, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--gamma', type=float, default=2, metavar='M',
                    help='learning rate decay factor (default: 0.5)')                   
parser.add_argument('--save', type=str, default='model_new.pt',
                    help='file on which to save model weights')
parser.add_argument('--GPUID', type=int, default=0,
                    help='file on which to save model weights')

args = parser.parse_args()

CONFIG = {"seed": 42,
          "img_size": 512,
          "model_name": "efficientnet",  # tf_efficientnet_b6_ns, tf_efficientnetv2_l_in21k, eca_nfnet_l2
          "num_classes": 15587,
          "embedding_size": 512,
          "train_batch_size": 10,
          "valid_batch_size": 10,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-6,
          "T_max": 29,
          "weight_decay": 0.0002,
          "n_fold": 2,
          "n_accumulate": 1,
          "gpu_parallel": False,
          "max_grad_norm": 1000,
          "amp": False,
          "num_workers": 10,
          "Modelname": "ArcFace_ColorJitter_new.pth",

          # ArcFace Hyperparameters
          "s": 30.0,  # arcface scale
          "m": 0.30,  # arcface margin
          "ls_eps": 0.0,  # arcface label smoothing
          "easy_margin": False,  # arcface easy_margin
          }


class HappyWhaleDataset(Dataset):
    def __init__(self,csv_file,root_dir, transform1= None, transform2= None):
        self.annotations =pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform1 = transform1
        self.transform2 = transform2
    def __len__(self):

        return len(self.annotations) #51033
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc[index,2])

        
        if self.transform1:
            image = self.transform1(image)
        image = image.type(torch.uint8)
        if self.transform2:
            image = self.transform2(image)
        
        return(image,y_label)


class HappyWhaleDataset1(Dataset):
    def __init__(self,csv_file,root_dir, transform= None):
        self.annotations =pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):

        return len(self.annotations) #51033
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(self.annotations.iloc[index,2])

        if self.transform:
            image = self.transform(image)
        
        
        return(image,y_label)

#Load data

mean2 = [0.4124,0.4564,0.5065]
std2 = [0.2245,0.2230,0.2336]

my_transform_fold2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ColorJitter(),
    transforms.Normalize(mean = [0.4124,0.4564,0.5065], std=[0.2245,0.2230,0.2336])
])

mean1 = [0.4108,0.4552,0.5053]
std1 = [0.2241,0.2227,0.2335]


my_transform_fold1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ColorJitter(),
    transforms.Normalize(mean1, std1)   
])


meannew = [0.4116,0.4558,0.5059]
stdnew = [0.2243,0.2229,0.2336]
my_transform_new = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ColorJitter(),
    transforms.Normalize(meannew, stdnew)
])
#Credit to
'''Cheng, S. (2022). GitHub - Sking-Cheng/Kaggle-Happywhale: Happywhale - Whale and Dolphin Identification SilverSolution (26/1588)
. Retrieved 7 June 2022, from https://github.com/Sking-Cheng/Kaggle-Happywhale'''
#code based on Cheng implementation
# Arcface
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """

    def __init__(self, in_features ,out_features,GPUID, s=30.0,
                 m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  # re-scale
        self.m = m  # margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)  # cos margin
        self.sin_m = math.sin(m)  # sin margin
        self.threshold = math.cos(math.pi - m)  # cos(pi - m) = -cos(m)
        self.mm = math.sin(math.pi - m) * m  # sin(pi - m)*m = sin(m)*m
        self.GPUID = GPUID

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))  
        phi = cosine * self.cos_m - sine * self.sin_m  # cosθ*cosm – sinθ*sinm = cos(θ + m)
        phi = phi.float()  # phi to float
        cosine = cosine.float()  # cosine to float
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # if cos(θ) > cos(pi - m) means θ + m < math.pi, so phi = cos(θ + m);
            # else means θ + m >= math.pi, we use Talyer extension to approximate the cos(θ + m).
            # if fact, cos(θ + m) = cos(θ) - m * sin(θ) >= cos(θ) - m * sin(math.pi - m)
            phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
            # https://github.com/ronghuaiyang/arcface-pytorch/issues/48
        # --------------------------- convert label to one-hot ---------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size()).cuda(self.GPUID)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HappyWhaleModel(nn.Module):
    def __init__(self, model_name, embedding_size,GPUID, pretrained=True):
        super(HappyWhaleModel, self).__init__()

        self.model = torchvision.models.efficientnet_b7(pretrained=pretrained)

        if 'efficientnet' in model_name:
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Identity()
            self.GPUID = GPUID


        #self.pooling = GeM()  # GeM Pooling ACA PODEMOS CAMBIAR POR EL MAX POOLING
        self.embedding = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, embedding_size)
        )
        # arcface
        self.fc = ArcMarginProduct(embedding_size,
                                   CONFIG["num_classes"],GPUID=GPUID,
                                   s=CONFIG["s"],
                                   m=CONFIG["m"],
                                   easy_margin=CONFIG["easy_margin"],
                                   ls_eps=CONFIG["ls_eps"])

    def forward(self, images, labels):
        '''
        train/valid
        '''
        features = self.model(images)
        
        #pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(features)  # embedding
        output = self.fc(embedding, labels)  # arcface
        return output

    def extract(self, images):
        '''
        test
        '''
        features = self.model(images)
        #pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(features)  # embedding
        return embedding


model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'],GPUID= args.GPUID )

# Descomentar esta parte si se quiere precargar un model del directorio
'''pretrained_dict = torch.load(os.path.join("arcface_AutoAugment_fold2.pth"))
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)'''

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)



def train_one_epoch(model, optimizer, scheduler, dataloader, epoch,GPUID):
    model.train().cuda(GPUID) #
    scaler = GradScaler()
    dataset_size = 0
    running_loss = 0.0
    pred_correct = 0
    counter = 0
    correct_top5 = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (data, target) in bar:

        images = data.cuda(GPUID) #
        labels = target.cuda(GPUID)

        batch_size = images.size(0)
        if CONFIG["amp"]:
            with autocast():
                outputs = model(images, labels).cuda(GPUID)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images, labels)
            
            loss = criterion(outputs, labels)

        loss = loss / CONFIG['n_accumulate']
        if CONFIG["amp"]:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            if CONFIG["amp"]:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        counter += len(labels)

        pred_correct += (np.argmax(outputs, -1) == labels).sum()
        acc = pred_correct / counter

        outputs_top5 = outputs.argsort()[:, -5:][:, ::-1]
        labels_top5 = np.expand_dims(labels, axis=1)
        correct_top5 += sum(np.any((labels_top5 == outputs_top5), axis=1))
        acc_top5 = correct_top5 / counter

        bar.set_postfix(Epoch=epoch,
                        Train_Loss=epoch_loss,
                        Train_Acc=acc,
                        Train_Top5_Acc=acc_top5,
                        grad_norm=grad_norm.item(),
                        LR=optimizer.param_groups[0]['lr']
                        )
    gc.collect()
    return epoch_loss

def valid_one_epoch(model, dataloader, GPUID, epoch):
    model.eval()
    
    dataset_size = 0 
    running_loss = 0.0 
    pred_correct = 0 
    counter = 0 
    correct_top5 = 0 
    bar = tqdm(enumerate(dataloader), total=len(dataloader)) 
    for step, (data,target) in bar:        
        images = data.cuda(GPUID)
        labels = target.cuda(GPUID) 
        
        batch_size = images.size(0)  

        outputs = model(images, labels).cuda(GPUID)
        loss = criterion(outputs, labels)
    
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

            
        outputs = outputs.detach().cpu().numpy()  
        labels = labels.detach().cpu().numpy() 
        counter += len(labels) 

        pred_correct += (np.argmax(outputs, -1) == labels).sum() 
        acc = pred_correct / counter 

        outputs_top5 = outputs.argsort()[:, -5:][:,::-1] 
        labels_top5 = np.expand_dims(labels, axis=1)
        correct_top5 += sum(np.any((labels_top5 == outputs_top5),axis=1)) 
        acc_top5 = correct_top5 / counter 
        
        bar.set_postfix(Epoch=epoch,
                        Valid_Loss=epoch_loss,
                        Valid_Acc=acc,
                        Valid_Top5_Acc=acc_top5,
                        LR=optimizer.param_groups[0]['lr'])   

    gc.collect()
    return epoch_loss 






def run_training(model, optimizer, scheduler, num_epochs, GPUID):

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, dataloader=train_loader, epoch=epoch, GPUID =GPUID)
        val_epoch_loss = valid_one_epoch(model, valid_loader, GPUID, epoch=epoch)

        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)



         # deep copy the model
        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})") 
            best_epoch_loss = val_epoch_loss 
            best_model_wts = copy.deepcopy(model.state_dict()) 
            PATH = "Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_loss, epoch)
            torch.save(model.state_dict(), str(CONFIG['Modelname'])) 
            # Save a model file from the current directory
            print(f"Model Saved")
        print()


    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60,
                                                                (time_elapsed % 3600) % 60))  # 打印本次训练的耗时
    print("Best Loss: {:.4f}".format(best_epoch_loss))  # 打印 best loss

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['T_max'], eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CONFIG['T_0'], eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None

    return scheduler



train_dataset = HappyWhaleDataset1(csv_file= "train.csv", root_dir=str(os.path.join("data1","train")),transform=my_transform_new )
train_loader = DataLoader(train_dataset, batch_size= args.train_batch_size,shuffle= True)

valid_dataset = HappyWhaleDataset1(csv_file= "val.csv", root_dir=str(os.path.join("data1","val")),transform=my_transform_new )
valid_loader = DataLoader(train_dataset, batch_size= args.val_batch_size,shuffle= True)

optimizer = optim.Adam(model.parameters(), lr= args.lr, weight_decay=CONFIG['weight_decay'])
scheduler = fetch_scheduler(optimizer) 
model, history = run_training(model, optimizer, scheduler, num_epochs=args.epochs,GPUID= CONFIG['GPUID'])