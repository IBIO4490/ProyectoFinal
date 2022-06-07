# Author: Nishanth Koganti
# Date: 2017/10/11

# Source: http://pytorch.org/tutorials/beginner/data_loading_tutorial.html

# Issues:
# Merge TrainDataset and TestDataset classes

# import libraries
##
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np

import glob

import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset

from skimage import io
from tqdm import tqdm

def Corregircsv():
    gt = pd.read_csv(os.path.join("train2.csv"))

    species = gt["species"]

    for i in tqdm(range(len(species))):

        if gt.loc[i,"species"] == "globis":
            gt.loc[i, "species"] = "short_finned_pilot_whale"
        elif gt.loc[i,"species"] == "pilot_whale":
            gt.loc[i, "species"] = "short_finned_pilot_whale"
        elif gt.loc[i,"species"] == "kiler_whale":
            gt.loc[i, "species"] = "killer_whale"
        elif gt.loc[i,"species"] == "bottlenose_dolpin":
            gt.loc[i, "species"] = "bottlenose_dolphin"


    gt.to_csv("train2.csv", index = False)
##
def ContarImgPorClase():
    gt = pd.read_csv(os.path.join("fold1.csv"))
    dicc = {}
    especie = gt["species"]

    for i in especie:
        if str(i) in dicc:
         dicc[str(i)] += 1

        else:
            dicc[str(i)] = 0
            dicc[str(i)] += 1
    return dicc
##
def ContarImgPorID(csv):
    gt = pd.read_csv(os.path.join(csv))
    dicc = {}
    especie = gt["individual_id"]

    for i in especie:
        if str(i) in dicc:
         dicc[str(i)] += 1

        else:
            dicc[str(i)] = 0
            dicc[str(i)] += 1
    return dicc
##
def ContadorDeImgPorIndividuo(imgsinf,imgssup):
    dicc = ContarImgPorID("train2.csv")
    contador1 = 0
    for i in range(15587):
        if dicc[str(i)] >= imgsinf and dicc[str(i)] <= imgssup :
            contador1+=1
    print(contador1)
## para ahcer la divison de las iamgenes, correr una vez solo
def Division2folds():

    gt = pd.read_csv(os.path.join("train2.csv"))
    image_name = gt["image"]
    species = gt["species"]
    prevalenciaxclase = ContarImgPorClase()

    os.mkdir(os.path.join("data", "fold1"))
    os.mkdir(os.path.join("data", "fold2"))

    # prevalencia vacia
    contador = {}
    clases = list(prevalenciaxclase.keys())
    for n in clases:
        contador[n] = 0

    for index in range(len(image_name)):
        especie_actual = species[index]

        if int(prevalenciaxclase[especie_actual]*0.5) >= contador[especie_actual]:
            contador[especie_actual] += 1

            shutil.move(os.path.join("data",str(image_name[index])), os.path.join("data", "fold1"))


    resto = glob.glob(os.path.join("data","*.jpg"))
    for j in range(len(resto)):
        shutil.move(resto[j], os.path.join("data", "fold2"))

## para ahcer la divison de las iamgenes, correr una vez solo 70, 15 ,15
def DivisionID(csv):

    gt = pd.read_csv(os.path.join("train2.csv"))
    image_name = gt["image"]
    id = gt["individual_id"]
    prevalenciaxid = ContarImgPorID(csv)

    os.mkdir(os.path.join("data", "train"))
    os.mkdir(os.path.join("data", "val"))
    os.mkdir(os.path.join("data", "test"))

    # prevalencia vacia
    contador = {}
    clases = list(prevalenciaxid.keys())
    for n in clases:
        contador[n] = 0
    listamovidas =[]
    for index in range(len(image_name)):
        id_actual = str(id[index])

        if prevalenciaxid[id_actual] != 1:
            if np.ceil(prevalenciaxid[id_actual] * 0.15) > contador[id_actual]:
                contador[id_actual] += 1
                shutil.move(os.path.join("data", str(image_name[index])), os.path.join("data", "val"))
                listamovidas.append(os.path.join("data", str(image_name[index])))

        '''if np.ceil(prevalenciaxid[id_actual]*0.15) >= contador[id_actual]:
            contador[id_actual] += 1
            shutil.move(os.path.join("data",str(image_name[index])), os.path.join("data", "train"))
            listamovidas.append(os.path.join("data",str(image_name[index])))'''

    for n in clases:
        contador[n] = 0

    for index in range(len(image_name)):
        id_actual = str(id[index])
        if prevalenciaxid[id_actual] != 1:
            if np.ceil(prevalenciaxid[id_actual]*0.15) > contador[id_actual]:
                pathimagen = os.path.join("data",str(image_name[index]))
                if pathimagen not in listamovidas:
                    contador[id_actual] += 1
                    shutil.move(os.path.join("data",str(image_name[index])), os.path.join("data", "test"))



    resto = glob.glob(os.path.join("data","*.jpg"))
    for j in range(len(resto)):
        shutil.move(resto[j], os.path.join("data", "train"))
##

def diccionario():
    gt = pd.read_csv(os.path.join("train2.csv"))
    species = gt["individual_id"]
    dicc = {}
    contador = 0
    for i in species:
        if i not in dicc:
            dicc[i] = contador
            contador += 1

    return dicc
## cambiar la clase por un numero de acuerdo al diccionario

def modificarcsv():
    gt = pd.read_csv(os.path.join("train2.csv"))
    dicc = diccionario

    species = gt["individual_id"]

    for i in tqdm(range(len(species))):
        NewValue = dicc[str(species[i])]
        gt.loc[i,"individual_id"] = NewValue

    gt.to_csv("train2.csv", index = False)

## crear los nuevos csv de ambos folds a partir del original
def CrearCSV(fold):
    gt = pd.read_csv(os.path.join("train2.csv"))
    Nombre = list(gt["image"])
    lista = glob.glob(os.path.join("data1",fold,"*.jpg"))
    listaNombres=[]
    ListaIndices = []
    for i in lista:
        listaNombres.append(i.split("\\")[-1])


    for j in tqdm(listaNombres):
        for k in range(len(Nombre)):

            if j == Nombre[k]:
                ListaIndices.append(k)
                continue
    dato = gt.iloc[ListaIndices]
    df = pd.DataFrame(data=dato)
    df.to_csv(fold+".csv", index=False)
##
class HappyWhalesDataset(Dataset):
    def __init__(self,csv_file,root_dir, transform= None):
        self.annotations =pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    def __len__(self):

        return len(self.annotations) #51033
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir,self.annotations.iloc[index,0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index,1]))

        if self.transform:
            image = self.transform(image)

        return(image,y_label)