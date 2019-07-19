### Simple routine to make a test-train-split for image classification in Keras / Tensorflow
### Generates .../train/... and .../val/...
### Original image folder names (in origdir) are used as class names

import os
from glob import glob
from shutil import copyfile

############################
# Data stored at...
odir = "C:/origdir/"
# Target dir for test-train split
tdir = "C:/myimages/"
# Test-train-split (= x * maxsamples)
trainsize = 0.8
# Define max numer of samples for test-train
maxsamples = 2500

paths = glob(str(odir)+"*")

for p in paths:
    # get name of dir
    classname = p[p.rfind("\\")+1:]
    ###########################################
    # Gen dirs in tt
    # Check/create dir
    # TEST
    try:
        os.makedirs(str(tdir) + "/val/" + str(classname))
    except FileExistsError:
        pass
    # TRAIN
    try:
        os.makedirs(str(tdir) + "/train/" + str(classname))
    except FileExistsError:
        pass
    # ###########################################
    # COPY
    # train samples
    filelist = os.listdir(p)
    filelist = filelist[:maxsamples]
    tindex = int(trainsize*len(filelist))
    trainfiles = filelist[:tindex]
    testfiles = filelist[tindex:]
    # train
    for f in trainfiles:
        copyfile(p + "/"+  f, str(tdir) + "/train/" + str(classname) + "/" + f)
        #os.remove(p + "/"+  f)
    # test
    for f in testfiles:
        # get filename
        copyfile(p + "/"+  f, str(tdir) + "/val/" + str(classname) + "/" + f)
        #os.remove(p + "/"+  f)
