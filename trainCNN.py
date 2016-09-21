import argparse,fileinput,os,sys,subprocess
caffe_root = '../caffe/' #PATH TO CAFFE ROOT
sys.path.insert(0,caffe_root + 'python')
import caffe
caffe.set_mode_cpu()



#Parse the arguments
def ParseInputArguments():
   parser = argparse.ArgumentParser()

   # Parse input arguments
   parser.add_argument('net', help = 'path to network architecture')
   parser.add_argument('train', help = 'path to training data')
   parser.add_argument('test', help = 'path to test data')
   parser.add_argument('snapshot_prefix', help = 'prefix of the output network')
   parser.add_argument('max_iter', type = int, help = 'total number of iterations')
   parser.add_argument('--init', help = 'path to pre-trained model')
   parser.add_argument('--init_type', choices = ['fin','res'], help = "fin: for finetuning, res: for resuming training")
   parser.add_argument('--base_lr', default = 0.01, type = float, help = 'initial learning rate')
   parser.add_argument('--display', default = 20, type = int, help = 'display output every #display iterations')
   parser.add_argument('--test_interval', default = 500 , type = int, help = 'test every #test_interval iterations')
   parser.add_argument('--snapshot', default = 500, type = int, help = 'produce an output every #snapshot iterations')
   parser.add_argument('--momentum',default = '0.9',  type = float, help = ' weight of the previous update')
   parser.add_argument('--lr_policy',default = 'step',choices=['step','fixed','exp','inv','multistep','poly','sigmoid'] ,help = 'learning rate decay policy')
   parser.add_argument('--test_iter', default = 75 , type = int, help = 'perform #test_iter iterations when testing')
   parser.add_argument('--stepsize', default = 700 , type = int, help = 'reduce learning rate every #stepsize iterations')
   parser.add_argument('--gamma',  default = 0.1, type = float, help = 'reduce learning rate to an order of #gamma')
   parser.add_argument('--weight_decay', default = 0.005, type = float, help = 'regularization term of the neural net')
   parser.add_argument('--solver_mode', default = 'CPU', choices = ['CPU','GPU'], help = 'where to run the program')
   parser.add_argument('--device_id', default = 0, type = int, choices=[0,1], help = '0:for CPU, 1: for GPU')
   args = parser.parse_args()
   solver = PrintSolverSetup(args)
   return args,solver

#Create the solver file .- Solver file works also as a descriptor for the experiment. -Extra information is written as comment
def PrintSolverSetup(args):   
   solver = args.snapshot_prefix+"_solver.prototxt"
   print "Export experiment parameters to solver file:",solver
   fsetup = open(args.snapshot_prefix+"_solver.prototxt", 'w')
   for arg in vars(args):
     print arg, getattr(args, arg)
     if arg in ['init','train','test','init_type']:
        fsetup.write('#')
     if (type(getattr(args, arg)) is str) and arg is not 'solver_mode':
	fsetup.write(arg +': "'+ str(getattr(args, arg))+'"\n')
	continue
     fsetup.write(arg +': '+ str(getattr(args, arg))+'\n')
   fsetup.write("test_state: { stage: 'test-on-test' }"+'\n')
   fsetup.write("test_initialization: false"+'\n')
   fsetup.write("random_seed: 1701")
   fsetup.close()
   return solver


#Change paths to training and  test data to the NETWORK.prototxt file
def ChangeNetworkDataRoots(train,test,ftrain,ftest):
   
   for line in fileinput.input(args.net, inplace=True):
   	tmp = line.split(':')
   	initstring = tmp[0]
   	if tmp[0].strip() =='phase':
   		phase = tmp[1].strip()
   	if tmp[0].strip() == 'source':
   		if phase.upper() == 'TRAIN':
   			print initstring+": \""+ftrain+"\"\n",
   		else:
   			print initstring+": \""+ftest+"\"\n",
   		continue
   	if tmp[0].strip() =='root_folder':
   		if phase.upper() == 'TRAIN':
   			print initstring+": \""+train+'/\"\n',
   		else:
   			print initstring+": \""+test+'/\"\n',
   		continue	
   	print line,	
   	   

	# Create Source Files
def  CreateResourceFiles(snapshot_prefix,train,test):

    allLabels = list(set(os.listdir(train)+os.listdir(test)))

    # Create Train Source File
    fnameTrain = snapshot_prefix + "_TrainSource.txt"
    train_file = open(fnameTrain, "w")
    for idx,label in enumerate(allLabels):
    	datadir = "/".join((train,label))
    	if os.path.exists(datadir):
           trainSamples = os.listdir(datadir)
           for sample in trainSamples:
        	   train_file.write('/'.join((label,sample))+' '+str(idx)+'\n')
    train_file.close()

    # Create Test Source File
    fnameTest = snapshot_prefix + "_TestSource.txt"
    test_file = open(fnameTest, "w")
    for idx,label in enumerate(allLabels):
    	datadir = "/".join((test,label))
    	if os.path.exists(datadir):
           testSamples = os.listdir(datadir)
           for sample in testSamples:
        	   test_file.write('/'.join((label,sample))+' '+str(idx)+'\n')
    test_file.close()
    return fnameTrain,fnameTest

# Modify execution file
def train(solver_prototxt_filename, init, init_type):
      for line in fileinput.input('train_net.sh', inplace=True):
              if '-solver' in line:
                 tmp = line.split('-solver')
                 if init==None:
                     print tmp[0]+" -solver "+ solver_prototxt_filename
                 elif init_type == 'fin':
                     print tmp[0]+" -solver "+ solver_prototxt_filename +" -weights " + init # .caffemodel file requiered for finetuning
                 elif init_type == 'res':
                     print tmp[0]+" -solver "+ solver_prototxt_filename +" -snapshot " + init # .solverstate file requiered for resuming training
                 else:
                     raise ValueError("No specific init_type defined for pre-trained network "+init)
              else:
                     print line,
      subprocess.call(["chmod", "+x","train_net.sh"])
      subprocess.call(['./train_net.sh'])



if __name__ == "__main__":
    args,solver = ParseInputArguments()
    ftrain,ftest = CreateResourceFiles(args.snapshot_prefix,args.train,args.test)
    ChangeNetworkDataRoots(args.train,args.test,ftrain,ftest)
    train(solver,args.init,args.init_type)
   
