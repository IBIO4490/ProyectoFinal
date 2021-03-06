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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Arguments
parser = argparse.ArgumentParser(description='PyTorch CNN HappyWhales')
parser.add_argument('--fold', type=str, default="test", metavar='N',
                    help='folder want to make test, ex: val or test (Default: test)')
parser.add_argument('--model', type=str, default='ArcFace_ColorJitter_new.pth',
                    help='model on which you want to take the test')
parser.add_argument('--img', type=str, default='0a0e4b82b9f3ee.jpg',
                    help='name of the image toe evaluate including the .jpg, we sure that the image you select is in the folder you indicate in args.fold (Default:0a0e4b82b9f3ee.jpg)')
parser.add_argument('--GPUID', type=int, default=1,
                    help='Gpu to run demo')
args = parser.parse_args()


CONFIG = {"seed": 42,
          "epochs": 10,
          "model_name": "efficientnet",  # tf_efficientnet_b6_ns, tf_efficientnetv2_l_in21k, eca_nfnet_l2
          "num_classes": 15587,
          "embedding_size": 512,
          "learning_rate": 1e-4,
          "scheduler": 'CosineAnnealingLR',
          "min_lr": 1e-4,
          "T_max": 9,
          "weight_decay": 0.0002,
          "n_fold": 2,
          "n_accumulate": 1,
          "gpu_parallel": False,
          "max_grad_norm": 1000,
          "amp": False,
          "num_workers": 10,

          # ArcFace Hyperparameters
          "s": 30.0,  # arcface scale
          "m": 0.30,  # arcface margin
          "ls_eps": 0.0,  # arcface label smoothing
          "easy_margin": False,  # arcface easy_margin
          }


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
        y_label = torch.tensor(int((self.annotations.loc[self.annotations['image'] == self.image])["individual_id"]))

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
        phi = cosine * self.cos_m - sine * self.sin_m  # cos??*cosm ??? sin??*sinm = cos(?? + m)
        phi = phi.float()  # phi to float
        cosine = cosine.float()  # cosine to float
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # if cos(??) > cos(pi - m) means ?? + m < math.pi, so phi = cos(?? + m);
            # else means ?? + m >= math.pi, we use Talyer extension to approximate the cos(?? + m).
            # if fact, cos(?? + m) = cos(??) - m * sin(??) >= cos(??) - m * sin(math.pi - m)
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
            #breakpoint()
        '''elif 'nfnet' in model_name:
            in_features = self.model.head.fc.in_features
            self.model.head.fc = nn.Identity()
            self.model.head.global_pool = nn.Identity()'''

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

test_dataset = HappyWhalesDatasetdemo(csv_file= "test.csv", root_dir=str(os.path.join("data1",str(args.fold))),transform=my_transform_test, image= args.img  )
test_loader = DataLoader(test_dataset, batch_size= 1,shuffle= False)

GPUID = args.GPUID

net = model.cuda(args.GPUID) # this will bring the network to GPU if DEVICE is cuda
net.eval() # Set Network to evaluation mode

dataset_size = 0
pred_correct = 0
counter = 0
correct_top5 = 0
for images, labels in test_loader:
    images = images.cuda(GPUID)
    labels = labels.cuda(GPUID)
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

    primerapred= np.argmax(outputs, -1)

    break

print("La predicci??n es " +str(primerapred) + " La anotaci??n es "+ str(labels))
print("Las 5 predicciones en orden son: " +str(outputs_top5))
'''if predsfinal == labelsfinal:
    print("La predicci??n fue correcta")
else:
    print("La predicci??n No fue correcta")'''


#test_epoch_loss = test_one_epoch(model, test_loader,GPUID)
