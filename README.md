# ProyectoFinal
* For Classification and Photo-ID you must have the folder data and data1 respectively, and their annotation files which are fold1.csv and fold2.csv for classfication
and test.csv,val.csv, train.csv, fold1ID.csv and fold2ID.csv for photo-ID. These are in the user Sojedaa/ProyectoFinal. All these aren't uploaded to the repository 
  as indicated in the instructions that the dataset will be taken from our users. The diccionaries named dicc_clases.pkl and dicc_id.pkl are important for the testing of the models so you can know exactly the species.
  
# Classification
*  You must pip install the last version of torchvision (0.12.0), 
   additionally you must install torch, pandas, numpy, math and time

  ```
  $ pip install torchvision==0.12.0
  ```

## Files

* [train.py](train.py): Script to train selected model.
* [test.py](test.py): Script to test models.
* [dataproc.py](trainer.py): Script with some functions that where used to adjust the dataset and annotations, and also correct some errors in the annotations.




## USAGE

* Test models by running `test.py` using the correct arguments:
  Ex:
  $ python3 test.py --mode test --fold 1 --CNN Resnet50 --model ResNet50_NoPre_Adam_lr1e-4_wd2e-4_Fold1.pth
  
the number of the fold must be different from the number of the model
  (Not all arguments are obligatory, they have default values)
  In the help description of each argument you can see a description. FOr the mode parameter put demo to test
  one image and test for all the dataset.
Take into account that in demo the argument --img must correspond in the folder selected.

# Photo-ID

## Files

* [TrainArcFace.py](TrainArcFace.py): Script to train selected model.
* [TestArcFace.py](TestArcFace.py): Script to test models.
* [DemoArcFace.py](TestArcFace.py): Script to evaluate one image.

## USAGE
* Take into account that as we made another modification to the dataset for our best result in photo-ID,
  here you won't need to select any folder, its just the test folder.
* Test models by running `TestArcFace.py` using the correct arguments:
  Ex:
  $ python3 TestArcFace.py --model ArcFace_ColorJitter_new.pth --GPUID 2
  
  (Not all arguments are obligatory, they have default values)

* evaluate one image by running `DemoArcFace.py` using the correct arguments:<
  Ex:
  $ python3 DemoArcFace.py --model ArcFace_ColorJitter_new.pth --GPUID 2 --img 0a0e4b82b9f3ee.jpg
  
  (Not all arguments are obligatory, they have default values)


