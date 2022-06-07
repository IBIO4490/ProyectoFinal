import gc
import math


# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from skimage import io
# Utils

from tqdm import tqdm

import torchvision.transforms as transforms
# Sklearn Imports

import argparse
import warnings

import os

warnings.filterwarnings("ignore")
# For descriptive error messages
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Arguments
parser = argparse.ArgumentParser(description='PyTorch Deeplab v3 Example')
parser.add_argument('--test_batch_size', type=int, default=10, metavar='N',
                    help='input batch size for testing (default: 10)')
parser.add_argument('--GPUID', type=int, default=2,
                    help='file on which to save model weights')
parser.add_argument('--model', type=str, default="ArcFace_ColorJitter_new.pth",
                    help='model to test')                    

args = parser.parse_args()



CONFIG = {"seed": 42,
          "model_name": "efficientnet", 
          "num_classes": 15587,
          "embedding_size": 512,

          # ArcFace Hyperparameters
          "s": 30.0,  # arcface scale
          "m": 0.30,  # arcface margin
          "ls_eps": 0.0,  # arcface label smoothing
          "easy_margin": False,  # arcface easy_margin
          }


class HappyWhaleDataset(Dataset):
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
        phi = cosine * self.cos_m - sine * self.sin_m  
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
            #self.model.avgpool = nn.Identity()

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


#torch.inference_mode()
def test_one_epoch(model, dataloader,GPUID):
    model.eval()
    model.cuda(GPUID)
    dataset_size = 0
    pred_correct = 0
    counter = 0
    correct_top5 = 0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, (data, target) in bar:
        images = data.cuda(GPUID)
        labels = target.cuda(GPUID)
        #breakpoint()
        batch_size = images.size(0)

        outputs = model(images, labels)
        
        
        dataset_size += batch_size
        

        outputs = outputs.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        counter += len(labels)

        pred_correct += (np.argmax(outputs, -1) == labels).sum()
        acc = pred_correct / counter

        outputs_top5 = outputs.argsort()[:, -5:][:, ::-1]
        labels_top5 = np.expand_dims(labels, axis=1)
        correct_top5 += sum(np.any((labels_top5 == outputs_top5), axis=1))
        acc_top5 = correct_top5 / counter

        

        bar.set_postfix(
                        Valid_Acc=acc,
                        Valid_Top5_Acc=acc_top5,
                        )
    gc.collect()
    
    print("Final accuracy: " + str(acc) + " Final Top 5 accuracy: "+ str(acc_top5))


model = HappyWhaleModel(CONFIG['model_name'], CONFIG['embedding_size'],GPUID= args.GPUID )

pretrained_dict = torch.load(os.path.join(args.model))
model_dict = model.state_dict()
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

#Load data

meannew = [0.4116,0.4558,0.5059]
stdnew = [0.2243,0.2229,0.2336]

my_transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.ColorJitter(),
    transforms.Normalize(meannew, stdnew)
])

test_dataset = HappyWhaleDataset(csv_file= "test.csv", root_dir=str(os.path.join("data1","test")),transform=my_transform_test )
test_loader = DataLoader(test_dataset, batch_size= args.test_batch_size,shuffle= False)

GPUID = args.GPUID

test_epoch_loss = test_one_epoch(model, test_loader,GPUID)
