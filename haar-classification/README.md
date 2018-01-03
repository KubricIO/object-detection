This is a tool to create haar-classifiers for on object using python-opencv.

Install step:
1. open this directory in terminal.
2. run "pip3 intall -r requirements.txt"

It needs to be supplied with images of the object(referred to as positive images) whose classifier is to be made. Care needs to be taken while choosing the images as if images contain other objects, it will affect the efficiency and accuracy of the classifier. The number of images needed to suppy depends upon the kind of object. For things which come in only one look, even a couple of images can be good enough. But for things like say a human face, you may need even hundreds.
It also needs to be supplied with background images(also called negative images).Care needs to b e taken here also. Ideally the images you supply should be similar to what you will be testing upoon. Too different backgrounds than what going to be tested upon can reduce accuracy. There should be atleast a 1000 images for a good classifier.

To create your classifier:
1. open this directory in terminal.
2. run "python3 classifier_create.py -h"
3. run "python3 classifier_create.py parameters"
4. give the required inputs (positive images folder path and negative images folder path, etc.)
	**Note the path to folders should be given as /home/username(i.e your username)/path



