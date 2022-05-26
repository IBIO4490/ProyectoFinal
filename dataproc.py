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
from PIL import Image
import glob
import random as rd
import shutil
import pandas as pd
import torch
from torch.utils.data import Dataset
import scipy.io
from skimage import io
from tqdm import tqdm
import pickle

## correccion:
#globis a short_finned_pilot_whale
#pilot_whale a short_finned_pilot_whale
#kiler_whale a killer_whale
#bottlenose_dolpin a bottlenose_dolphin

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
Corregircsv()
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
ContarImgPorClase()
## para ahcer la divison de las iamgenes, correr una vez solo
def Division():

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

##

def diccionario():
    gt = pd.read_csv(os.path.join("train2.csv"))
    species = gt["species"]
    dicc = {}
    contador = 0
    for i in species:
        if i not in dicc:
            dicc[i] = contador
            contador += 1

    return dicc
##para guardar el diccionario, se corre una sola vez
diccionario = diccionario()
pickle_out = open("dicc_clases.pkl", "wb")
pickle.dump(diccionario,pickle_out)
pickle_out.close()
diccionario1 = pickle.load(open("dicc_clases.pkl","rb"))

## cambiar la clase por un numero de acuerdo al diccionario

def modificarcsv():
    gt = pd.read_csv(os.path.join("train2.csv"))
    dicc = diccionario1

    species = gt["species"]

    for i in tqdm(range(len(species))):
        NewValue = dicc[str(species[i])]
        gt.loc[i,"species"] = NewValue

    gt.to_csv("train2.csv", index = False)
modificarcsv()
## crear los nuevos csv de ambos folds a partir del original
def CrearCSV(fold):
    gt = pd.read_csv(os.path.join("train2.csv"))
    Nombre = list(gt["image"])
    lista = glob.glob(os.path.join("data",fold,"*.jpg"))
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


CrearCSV("fold1")
# ##
# gt = pd.read_csv(os.path.join("train2.csv"))
# #gt = gt.drop(0)
# print(gt.iloc[[0]])
# lista = [0,1,2,3,4]
# dato = gt.iloc[lista]
# ##
# df = pd.DataFrame(data = dato)
# df.append(gt.iloc[[1]], ignore_index = True)
# print(df)
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

