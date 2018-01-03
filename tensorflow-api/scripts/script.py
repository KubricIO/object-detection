import shutil,sys,os,subprocess,glob,argparse,re,fileinput

cwd = os.getcwd()

def prepare_config(config_file,n,bs):
	with open(config_file,'r') as file:
		file=file.read().split("\n")
		f=open('training/myconfig.config',"w+")
		i=5
		while i < len(file):
			line=file[i]
			if re.search(r'num_classes(.*)',line):
				f.write("    num_classes: "+str(n)+"\n")
			elif re.search(r'fine_tune_checkpoint',line):
				f.write('  fine_tune_checkpoint: "config_dir/model.ckpt"\n')
			elif re.search(r'batch_size',line):
				f.write('  batch_size: '+str(bs)+'\n')
			elif re.search(r'train_input_reader',line):
				s='train_input_reader: {\n  tf_record_input_reader {\n    input_path: "data/train.record"\n  }'
				s=s+'\n  label_map_path: "training/object-detection.pbtxt"\n}\n\neval_config: {\n  num_examples:40\n'
				f.write(s)
				i+=8
			elif re.search(r'eval_input_reader',line):
				f.write(line+"\n  ")
				s='tf_record_input_reader {\n    input_path: "data/test.record"\n  }\n'
				s=s+'  label_map_path: "training/onject-detection.pbtxt"\n'
				f.write(s)
				i+=4
			else:
				f.write(line+"\n")
			i+=1
		f.close()

def prepare(train_img_dir,test_img_dir,ckpt_dir):
	if not os.path.exists(cwd+'/images'):
		os.makedirs('images')
	if not os.path.exists(cwd+'/config_dir'):
		os.makedirs('config_dir')
	try:
		shutil.copy(train_img_dir,cwd+"/images")
	except:
		print('SameFileError')
	try:
		shutil.copy(test_img_dir,cwd+"/images")
	except:
		print('SameFileError')
	for filename in glob.glob(os.path.join(ckpt_dir, '*.*')):
		shutil.copy(filename,cwd+"/config_dir")

def create_tfrecord(obj_list,train_csv_path="",test_csv_path="",from_xml=True):
	if from_xml :
		try:
			process="python3 xml_to_csv.py"
			os.system(process)
		except:
			process="python xml_to_csv.py"
			os.system(process)
	else:
		try:
			shutil.copy(train_csv_path,cwd+"/data")
		except:
			print('already present')
		os.chdir("data")
		os.rename(train_csv_path.split("/")[-1],"train_labels.csv")
		try:
			shutil.copy(test_csv_path,cwd+"/data")
		except:
			print('already present')
		os.rename(test_csv_path.split("/")[-1],"test_labels.csv")
		os.chdir("..")
	for f in ['train','test']:
		process="generate_tfrecord.py --csv_input=data/"+f+"_labels.csv  --output_path=data/"+f+".record"
		process=process+" --obj_list="+obj_list
		try:
			os.system("python3 "+process)
		except ImportError:
			os.system("python "+process)
		except :
			print("ImportError: No module named tensorflow")

def create_pbtxt(obj_list):
	f=open("training/object-detection.pbtxt","w+")
	for i,obj in enumerate(obj_list):
		seg="item {\n  id: "+str(i+1)+"\n  name: '"+str(obj)+"'\n}"
		f.write(seg+"\n")

if __name__=='__main__':
	p=argparse.ArgumentParser()
	p.add_argument('train_img_dir',help="directory containing images to be trained")
	p.add_argument('test_img_dir',help="directory containing images to be tested")
	p.add_argument('train_csv_path',help="path to the csv file containing annotaions of training images")
	p.add_argument('test_csv_path',help="path to the csv file containing annotaions of testing images")
	p.add_argument('ckpt_dir',help="directory containg the checkpoint files for the configuration you are using")
	p.add_argument('--from_xml',help="whether you have .xml files for creating the tfrecord")
	p.add_argument('config_file',help="the configuration file")
	p.add_argument('obj_list',help="list of your objects' names separated by '/'. Example: object_1/object_2/..../object_n")
	p.add_argument('batch_size',help="the size of the batch which is going to be used in one step of the training")
	a=p.parse_args()
	os.chdir("..")
	pwd=os.getcwd()
	sys.path.insert(0,pwd)
	sys.path.insert(0,pwd+"/slim")
	os.chdir("./object_detection")
	prepare(a.train_img_dir,a.test_img_dir,a.ckpt_dir)
	create_tfrecord(a.obj_list,train_csv_path=a.train_csv_path,test_csv_path=a.test_csv_path,from_xml=(a.from_xml=='True'))
	prepare_config(a.config_file,len(a.obj_list),a.batch_size)
	create_pbtxt(a.obj_list.split('/'))
	process="train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/myconfig.config"
	try:
		os.system("python3 "+process)
	except:
		os.system("python "+process)
