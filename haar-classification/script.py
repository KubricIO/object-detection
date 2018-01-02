import shutil,sys,os,subprocess,glob,argparse,re
from shutil import *
import mergevec as mv
import resize as rs
import tester as ts

cwd = os.getcwd() #current working directory
folders = ['pos_train_images','pos_test_images','neg_train_images',
    'neg_test_images','positive_images','negative_images','classifier','samples']


#copy all files from one directory to another
def copy_all(source_dir,dest_dir):
    for filename in glob.glob(os.path.join(source_dir, '*.*')):
        shutil.copy(filename, dest_dir)


# move n files from one directory to another
def move_some(source_dir,dest_dir,n):
    s_dir = os.listdir(source_dir)
    for i,file in enumerate(s_dir):
        if(i>n):
            break
        if not os.path.exists(dest_dir+"/"+file):
            shutil.move(source_dir+"/"+file,dest_dir)


#create .txt files for positive and negative images
def create_txt(path,f_name):
    f= open(f_name,"w+")
    files = os.listdir(path)
    for file in files:
        f.write(path + "/" + file+"\n")


#remove all files with given extension from a directory
def rem_ext(dir, ext):
    for f in os.listdir(dir):
        if re.search(ext, f):
            os.remove(os.path.join(dir, f))


#prepare the directories and files for execution
def prepare(pfolder,nfolder,h,w,num_train):
    global folders
    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder,ignore_errors=False,onerror=None)
    rem_ext(cwd,".txt")
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    folders = [cwd+"/"+f for f in folders]
    copy_all(pfolder,folders[1])
    copy_all(nfolder,folders[3])
    move_some(folders[1],folders[0],num_train)
    move_some(folders[3],folders[2],1000)
    rs.resize_all(folders[0],h*10,w*10,folders[4])
    rs.resize_all(folders[2],400,400,folders[5])
    create_txt(folders[4],"positives.txt")
    create_txt(folders[5],"negatives.txt")


#create samples for the training
def create_samples(num_samples,h,w):
    global folders
    foldr = os.listdir(folders[4])
    num=str(int(num_samples)/(len(foldr)-1))
    for file in foldr:
        p_str = "-img "+folders[4]+"/"+file+" -bg negatives.txt -info in.txt -maxxangle 0.1 -maxyangle 0.1 -maxzangle 0.2 -maxidev 40 -num "+num
        args = ["opencv_createsamples"] + p_str.split(" ")
        pipe = subprocess.call(args)
        p_str = "-info in.txt -num "+num+" -w "+str(w)+" -h "+str(h)+" -vec "+folders[7]+"/"+file+".vec"
        args = ["opencv_createsamples"] + p_str.split(" ")
        pipe = subprocess.call(args)
        rem_ext(cwd,".jpg")
    mv.merge_vec_files(folders[7],'samples.vec')


#train the classifier with the samples
def train_samples(num_samples,h,w,num_neg):
    global folders
    o_str = """-data classifier -vec samples.vec -bg negatives.txt\
    -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos """+str(num_samples)+"""\
    -numNeg """+str(num_neg)+" -w "+str(w)+" -h "+str(h) +""" -mode ALL -precalcValBufSize 1024\
    -precalcIdxBufSize 1024 -featureType LBP"""
    args=["opencv_traincascade"]+o_str.split()
    pipe = subprocess.call(args)


#test the generated classifier
def test(name,min_size,scale_factor,min_neighbours):
    name = name+"_cascade.xml"
    global folders
    rem_ext(folders[1],".txt")
    rem_ext(folders[3],".txt")
    ts.tester(cwd+"/"+name,folders[1],folders[3],min_size,scale_factor,min_neighbours)


#create the cascade file
def create(name,pfolder,nfolder,ratio,num_samples,num_train):
    name = name+"_cascade.xml"
    neg_dir=os.listdir(nfolder)
    num_neg = min(num_samples/2,len(neg_dir))
    num_samples = str(num_samples)
    n=40
    if(ratio>1):
        w = int(n)
        h = int(ratio*n)
    else:
        h = int(n)
        w = int(n/ratio)

    prepare(pfolder,nfolder,h,w,num_train)
    create_samples(num_samples,h,w)
    train_samples(num_samples,h,w,num_neg)
    shutil.move(folders[6]+"/cascade.xml",cwd+"/")
    os.rename("cascade.xml",name)


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
    create(a.name,a.pfolder,a.nfolder,float(a.hw_ratio),int(a.num_samples),int(a.num_train))
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
    test(a.name,min_size,scale_factor,min_neighbours)
