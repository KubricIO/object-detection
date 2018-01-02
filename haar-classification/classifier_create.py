import os,glob,argparse
import shutil
from shutil import *


cwd = os.getcwd() #current working directory


if __name__=='__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('name',help="The name of the object whose classifier is to be created")
    parser.add_argument('pfolder',help="path to Folder containing images of the object(positive images)")
    parser.add_argument('nfolder',help="path to Folder containing images not containing the object(negative images)")
    parser.add_argument('hw_ratio',help="height to width ratio of object")
    parser.add_argument('num_samples', help="number of samples to be used for training the classifier")
    parser.add_argument('num_train',help="number of positive images used to train the classifier")
    parser.add_argument('--ms',help=""""minimum size of the object(in a pair of pixels) to be detected(example : (100,100)). 
        This is important during testing as there can small patterns in the test images that can resemble with those in the original images of the object, but are not really the objects.
        Its good to keep it a little big say atleast (40,40)""")
    parser.add_argument('--sf',help="Scale Factor, specifying how much the image size is reduced at each image scale.")
    parser.add_argument('--mn',help="MinNeighbours, specifying how many neighbors each candidate rectangle should have to retain it.")
    a= parser.parse_args()
    if not os.path.exists(a.name):
        os.makedirs(a.name)
    files = glob.iglob(os.path.join(cwd, "*.py"))
    for file in files:
    	if os.path.isfile(file):
        	shutil.copy2(file,cwd+"/"+a.name)
    os.chdir(cwd+"/"+a.name)
    cwd=os.getcwd()
    import script
    script.create(a.name,a.pfolder,a.nfolder,float(a.hw_ratio),int(a.num_samples),int(a.num_train))
    min_size=(100,100)
    scale_factor=1.2
    min_neighbours=6
    if a.ms:
        a.ms=a.ms.split(",")
        min_size=tuple([int(x) for x in a.ms])
    if a.sf:
        scale_factor=float(a.sf)
    if a.mn:
        min_neighbours=int(a.mn)
    script.test(a.name,min_size,scale_factor,min_neighbours)