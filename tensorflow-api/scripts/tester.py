import numpy as np
import os,shutil
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import re
import zipfile
import argparse
import IPython
import export_inference_graph as eig
from IPython import get_ipython
from collections import defaultdict
from io import StringIO
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image

cwd=os.getcwd()
sys.path.append("..")
matplotlib.use('Agg')

from utils import label_map_util
from utils import visualization_utils as vis_util

def update(file,m,s):
	matchObj=re.match(r'model.ckpt-(.*).'+s,file)
	if matchObj:
		return max(m,int(matchObj.group(1)))
	else:
		return m

if __name__=="__main__":
	p=argparse.ArgumentParser()
	p.add_argument("num_obj",help="number of objects")
	p.add_argument("num_img",help="number of test images")
	a=p.parse_args()
	if os.path.exists("objects_inference_graph"):
            shutil.rmtree("objects_inference_graph",ignore_errors=False,onerror=None)
	if not os.path.exists(cwd+"/objects_inference_graph"):
		os.makedirs("objects_inference_graph")
	if not os.path.exists(cwd+"/Results"):
		os.makedirs("Results")

	lis=os.listdir("training")
	maxm=0
	m1=0
	m2=0
	m3=0
	for file in lis:
		m1=update(file,m1,'meta')
		m2=update(file,m2,'index')
		m3=update(file,m3,'data')
		if m1==m2 and m2==m3:
			maxm=m1

	for i,file in enumerate(os.listdir("test_images")):
		os.rename("test_images/"+file,"test_images/img"+str(i)+".jpg")
	for i,file in enumerate(os.listdir("test_images")):
		os.rename("test_images/"+file,"test_images/image"+str(i)+".jpg")

	# tf.app.run(eig.main)
	process="export_inference_graph.py --input_type image_tensor --pipeline_config_path training/myconfig.config "
	process=process+"--trained_checkpoint_prefix training/model.ckpt-"+str(maxm)+" --output_directory objects_inference_graph"
	try:
		os.system("python "+process)
	except:
		os.system("python3 "+process)


	MODEL_NAME = 'objects_inference_graph'
	PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
	PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')
	NUM_CLASSES = int(a.num_obj)

	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')

	label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
	category_index = label_map_util.create_category_index(categories)

	def load_image_into_numpy_array(image):
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	PATH_TO_TEST_IMAGES_DIR = 'test_images'
	TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, int(a.num_img)+1) ]

	# Size, in inches, of the output images.
	IMAGE_SIZE = (12, 8)

	with detection_graph.as_default():
		with tf.Session(graph=detection_graph) as sess:
			# Definite input and output Tensors for detection_graph
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			# Each box represents a part of the image where a particular object was detected.
			detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			# Each score represent how level of confidence for each of the objects.
			# Score is shown on the result image, together with the class label.
			detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
			detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			n=0
			img_dict={}
			for image_path in TEST_IMAGE_PATHS:
				n+=1
				image = Image.open(image_path)
				# the array based representation of the image will be used later in order to prepare the
				# result image with boxes and labels on it.
				image_np = load_image_into_numpy_array(image)
				# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
				image_np_expanded = np.expand_dims(image_np, axis=0)
				# Actual detection.
				(boxes, scores, classes, num) = sess.run(
					[detection_boxes, detection_scores, detection_classes, num_detections],
					feed_dict={image_tensor: image_np_expanded})
				# Visualization of the results of a detection.
				vis_util.visualize_boxes_and_labels_on_image_array(
					image_np,
					np.squeeze(boxes),
					np.squeeze(classes).astype(np.int32),
					np.squeeze(scores),
					category_index,
					use_normalized_coordinates=True,
					line_thickness=8)
				plt.figure(figsize=IMAGE_SIZE)
				plt.imsave('Results/'+str(n)+'.png',image_np)
				lis=[category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
				for dic in lis:
					if dic['name'] not in img_dict:
						img_dict[dic['name']]=0
					else:
						img_dict[dic['name']]+=1
			print(img_dict)