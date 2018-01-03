This is a tool helps to use the object-detection api of tensorflow with reduced effort.

The first thing to do is to clone the tensorlfow/models repository from github.
You may do it with "git clone https://github.com/tensorflow/models.git" from terminal. Then paste all the .py files in the scripts directory into models/research/object_detection.

Now tensorlfow and other python packages need to be installed. You may refer to models/research/object_detection/g3doc/installation.md for this or try the following: 
1. open this models/research/object_detection in terminal.
2. run 'sudo pip2 install -r requirements.txt'
3. run 'sudo pip3 install -r requirements.txt'
(You can use either of the two above or both, depending on whether you installed tensorflow with python2 or python3. Its better to install tensorflow-gpu)

The preparation step involves annotation of images. All the images used for training and testing should be annotated in .xml format(which can then be converted into .csv) or directly in .csv format. For image annotation you may use labelImg which saves the results in .xml format. If you are using .xml files you must put it in the same directory as the images.
It also needs a configuration file which contains various parameters specified according to your need. There are some cofiguration files present in the /models/research/object_detection/samples/ directory one of which can be used. You also need checkpoint files for thr config you have chosen. You can find them on the web. One is here for an example, http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz (Extract it). Now we are ready. You can now do the following:

To train your model(from models/research/object_detection directory):
1.run "python(or python3) script.py -h".(You can see what inputs you need to provide)
2.run "python(or python3) script.py [arguments]"
3.Wait! for until the Loss starts to stay in the range 0.7-1.0 at most of the steps. It generally takes thousands of steps.
You can find varios .ckpt files in the trainig directory.

To test your model:
1.run "python(or python3) tester.py -h.
2.run "python(or python3) tester.py [arguments]
You can see some boxes(look at the few images ex$y.png where y=1,2,3) on the images you provided for testing in the Results directory.

In case of any problem you may refer to the tensorflow object-detection tutorial at https://pythonprogramming.net/introduction-use-tensorflow-object-detection-api-tutorial/.
