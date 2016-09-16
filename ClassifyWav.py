
from scipy.io import loadmat
from pyAudioAnalysis import audioBasicIO as io, audioFeatureExtraction as aF, audioSegmentation
import hmmlearn.hmm
import sys, os, glob
import cPickle
import random
import string
import numpy as np, scipy, matplotlib, Image
import matplotlib.pyplot as plt
import time

#Load Caffe library
caffe_root = '../../'
sys.path.insert(0,caffe_root + 'python')
import caffe
caffe.set_mode_cpu()

global RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB 

def initialize_transformer(image_mean, is_flow):
  shape = (10*16, 3, 227, 227)
  transformer = caffe.io.Transformer({'data': shape})
  channel_mean = np.zeros((3,227,227))
  for channel_index, mean_val in enumerate(image_mean):
    channel_mean[channel_index, ...] = mean_val
  transformer.set_mean('data', channel_mean)
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2, 1, 0))
  transformer.set_transpose('data', (2, 0, 1))
  transformer.set_is_flow('data', is_flow)
  return transformer

def singleFrame_classify_video(signal, net, transformer, with_smoothing):
    batch_size = 1 
    input_images = []

    input_im = caffe.io.load_image(signal.replace(".wav",".png"))        
    input_images.append(input_im)
    os.remove(signal.replace(".wav",".png"))    
    #Initialize predictions matrix                
    output_predictions = np.zeros((len(input_images),2))
    output_classes = []

    for i in range(0,len(input_images)):        
        # print "Classifying Spectrogram: ",i+1         
        clip_input = input_images[i:min(i+batch_size, len(input_images))] #get every image -- batch_size==1
        clip_input = caffe.io.oversample(clip_input,[227,227]) #make it 227x227        
        caffe_in = np.zeros(np.array(clip_input.shape)[[0,3,1,2]], dtype=np.float32) #initialize input matrix
        for ix, inputs in enumerate(clip_input):
            caffe_in[ix] = transformer.preprocess('data',inputs) # transform input data appropriatelly and add to input matrix        
        net.blobs['data'].reshape(caffe_in.shape[0], caffe_in.shape[1], caffe_in.shape[2], caffe_in.shape[3]) #make input caffe readable        
        out = net.forward_all(data=caffe_in) #feed input to the network
        output_predictions[i:i+batch_size] = np.mean(out['probs'].reshape(10,caffe_in.shape[0]/10,2),0) #predict labels        
        
        #Store predicted Labels without smoothing
        if  output_predictions[i:i+batch_size].argmax(axis=1)[0] == 0:
            prediction = "music"
        else:
            prediction = "speech"
        output_classes.append(prediction)
        #print "Predicted Label for file -->  ", signal.upper() ,":",    prediction
    return output_classes, output_predictions

def mtCNN_classification(signal, Fs, mtWin, mtStep, RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB):
    mtWin2 = int(mtWin * Fs)
    mtStep2 = int(mtStep * Fs)
    stWin = 0.020
    stStep = 0.015
    classesAll =["music", "speech"]
    N = len(signal)
    curPos = 0
    count = 0
    fileNames = []
    flagsInd = []
    Ps = []
    randomString = (''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5)))
    while (curPos < N):                 # for each mid-term segment
        N1 = curPos
        N2 = curPos + mtWin2 + stStep*Fs
        if N2 > N:
            N2 = N
        xtemp = signal[int(N1):int(N2)]                # get mid-term segment        

        specgram, TimeAxis, FreqAxis = aF.stSpectogram(xtemp, Fs, round(Fs * stWin), round(Fs * stStep), False)     # compute spectrogram
        if specgram.shape[0] != specgram.shape[1]:                                                                  # TODO (this must be dynamic!)
            break
        specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')        # resize to 227 x 227
        
        imSpec = Image.fromarray(np.uint8(matplotlib.cm.jet(specgram)*255))                                         # create image
        curFileName = randomString + "temp_{0:d}.png".format(count)
        fileNames.append(curFileName)    
        scipy.misc.imsave(curFileName, imSpec)
        
        T1 = time.time()
        output_classes, outputP = singleFrame_classify_video(curFileName, RGB_singleFrame_net, transformer_RGB, False)        
        T2 = time.time()
        #print T2 - T1
        flagsInd.append(classesAll.index(output_classes[0]))
        Ps.append(outputP[0])
        #print flagsInd[-1]
        curPos += mtStep2               
        count += 1              
    return np.array(flagsInd), classesAll, np.array(Ps)

def loadCNN():
    singleFrame_model = 'SOUND_deploy_singleFrame.prototxt'
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_1000_ALL_TRAIN_iter_1000.caffemodel'
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_2000_ALL_TRAIN_augmented_iter_2000.caffemodel' # augmented
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_4000_iter_augmented.caffemodel' # augmented
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_6000_iter_augmented.caffemodel' # augmented
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_10000_iter_augmented.caffemodel' # augmented
    
    # no fine tune (trained from scratch)
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_no_finetune_original_data_iter_1000.caffemodel' 
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_no_finetune_original_data_iter_500.caffemodel'  
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_no_finetune_original_data_iter_1500.caffemodel'   
    #RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_no_finetune_original_data_iter_2000.caffemodel'   # RUNNING 1

    # finetuned from imagenet
    RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_with_finetune_original_data_iter_500.caffemodel' # RUNNING 3
    # RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_with_finetune_original_data_iter_1000.caffemodel' # TODO
    # RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_with_finetune_original_data_iter_1500.caffemodel' # TODO
    # RGB_singleFrame = 'SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_with_finetune_original_data_iter_2000.caffemodel' # RUNNING 2

    RGB_singleFrame_net =  caffe.Net(singleFrame_model, RGB_singleFrame, caffe.TEST)
                
    #Mean RGB values
    SOUND_mean_RGB = np.zeros((3,1,1))
    SOUND_mean_RGB[0,:,:] = 103.939
    SOUND_mean_RGB[1,:,:] = 116.779
    SOUND_mean_RGB[2,:,:] = 128.68

    #INitialize input image transformer
    transformer_RGB = initialize_transformer(SOUND_mean_RGB, False)

    return RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB

def computePreRec(CM, classNames):
    numOfClasses = CM.shape[0]
    if len(classNames) != numOfClasses:
        print "Error in computePreRec! Confusion matrix and classNames list must be of the same size!"
        return
    Precision = []
    Recall = []
    F1 = []    
    for i, c in enumerate(classNames):
        Precision.append(CM[i,i] / np.sum(CM[:,i]))
        Recall.append(CM[i,i] / np.sum(CM[i,:]))
        F1.append( 2 * Precision[-1] * Recall[-1] / (Precision[-1] + Recall[-1]))
    return Recall, Precision, F1


def trainMetaClassifier(dirName, outputmodelName, modelName, method = "svm", postProcess = 0, PLOT = False):        
    types = ('*.wav', )
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))    
    wavFilesList = sorted(wavFilesList)    
    flagsAll = np.array([])

    for ifile, wavFile in enumerate(wavFilesList):                                                                          # for each wav file in folder
        print "{0:s}, {1:d} file of {2:d}".format(wavFile, ifile+1, len(wavFilesList))                                
        matFile = wavFile.replace(".wav","_true.mat")                                                                       # load current ground truth
        if os.path.isfile(matFile):
            matfile = loadmat(matFile)
            segs_gt = matfile["segs_r"]
            classes_gt1 = matfile["classes_r"]            
            classes_gt = []
            for c in classes_gt1[0]:
                if c == "M":
                    classes_gt.append("music")
                if c == "S" or c=="E":
                    classes_gt.append("speech")            
            flagsIndGT, classesAllGT = audioSegmentation.segs2flags([s[0] for s in segs_gt], [s[1] for s in segs_gt], classes_gt, 1.0)            
        #if method == "svm":
            # speech-music segmentation:
        #    [flagsInd, classesAll, acc] = audioSegmentation.mtFileClassification(fileName, modelName, "svm", False, '')            
        if method == "cnn":                                                                                                     # apply the CNN on the current WAV
            WIDTH_SEC = 2.4    
            [Fs, x] = io.readAudioFile(wavFile)                                                                                 # read the WAV
            x = io.stereo2mono(x)
            [flagsInd, classesAll, P] = mtCNN_classification(x, Fs, WIDTH_SEC, 1.0, RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB)    #  apply the CNN mid-term classifier       
            print len(flagsIndGT), P.shape                                                                                        # append the current ground truth labels AND estimated probabilities (either from the CNN or the SVM) on the global arrays

            lenF = P.shape[0]
            lenL = len(flagsIndGT)
            MIN = min(lenF, lenL)
            P = P[0:MIN, :]
            flagsIndGT = flagsIndGT[0:MIN]

            flagsNew = []
            for j, fl in enumerate(flagsIndGT):      # append features and labels
                flagsNew.append(classesAll.index(classesAllGT[flagsIndGT[j]]))

            flagsAll = np.append(flagsAll, np.array(flagsNew))
            
            if ifile == 0:
                Fall = P
            else:
                Fall = np.concatenate((Fall, P), axis=0)

            print Fall.shape
            print flagsAll.shape

    startprob, transmat, means, cov = audioSegmentation.trainHMM_computeStatistics(Fall.T, flagsAll)        # compute HMM statistics
    hmm = hmmlearn.hmm.GaussianHMM(startprob.shape[0], "diag")                          # train HMM
    hmm.startprob_ = startprob
    hmm.transmat_ = transmat        
    hmm.means_ = means
    hmm.covars_ = cov

    fo = open(outputmodelName, "wb")   # save HMM model
    cPickle.dump(hmm, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(classesAll, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    fo.close()

    return hmm, classesAll


def trainHMM(dirName, outputmodelName):        
    types = ('*.wav', )
    wavFilesList = []
    for files in types:
        wavFilesList.extend(glob.glob(os.path.join(dirName, files)))    
    wavFilesList = sorted(wavFilesList)    
    flagsAll = np.array([])
    mtWin = 3.0
    mtStep = 1.0
    classesAll =["music", "speech"]
    for ifile, wavFile in enumerate(wavFilesList):                                                                          # for each wav file in folder
        print "{0:s}, {1:d} file of {2:d}".format(wavFile, ifile+1, len(wavFilesList))                                
        matFile = wavFile.replace(".wav","_true.mat")                                                                       # load current ground truth
        if os.path.isfile(matFile):
            matfile = loadmat(matFile)
            segs_gt = matfile["segs_r"]
            classes_gt1 = matfile["classes_r"]            
            classes_gt = []
            for c in classes_gt1[0]:
                if c == "M":
                    classes_gt.append("music")
                if c == "S" or c=="E":
                    classes_gt.append("speech")            
            flagsIndGT, classesAllGT = audioSegmentation.segs2flags([s[0] for s in segs_gt], [s[1] for s in segs_gt], classes_gt, 1.0)            

            [Fs, x] = io.readAudioFile(wavFile)                                                                  # read the WAV
            [F, _] = aF.mtFeatureExtraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * 0.050), round(Fs * 0.050))     # feature extraction

            flagsNew = []
            for j, fl in enumerate(flagsIndGT):      # append features and labels
                flagsNew.append(classesAll.index(classesAllGT[flagsIndGT[j]]))


            lenF = F.shape[1]
            lenL = len(flagsNew)
            MIN = min(lenF, lenL)
            F = F[0:MIN, :]
            flagsNew = flagsNew[0:MIN]

            flagsAll = np.append(flagsAll, np.array(flagsNew))
            
            if ifile == 0:
                Fall = F
            else:
                Fall = np.concatenate((Fall, F), axis=1)

            print Fall.shape
            print flagsAll.shape

    startprob, transmat, means, cov = audioSegmentation.trainHMM_computeStatistics(Fall, flagsAll)        # compute HMM statistics
    hmm = hmmlearn.hmm.GaussianHMM(startprob.shape[0], "diag")                                            # train HMM
    hmm.startprob_ = startprob
    hmm.transmat_ = transmat        
    hmm.means_ = means
    hmm.covars_ = cov

    fo = open(outputmodelName, "wb")   # save HMM model
    cPickle.dump(hmm, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(classesAll, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mtWin, fo, protocol=cPickle.HIGHEST_PROTOCOL)
    cPickle.dump(mtStep, fo, protocol=cPickle.HIGHEST_PROTOCOL)    
    fo.close()

    return hmm, classesAll


def evaluateSpeechMusic(fileName, modelName, method = "svm", postProcess = 0, postProcessModelName = "", PLOT = False):        
        # load grount truth file (matlab annotation)

        matFile = fileName.replace(".wav","_true.mat")    
        if os.path.isfile(matFile):
            matfile = loadmat(matFile)
            segs_gt = matfile["segs_r"]
            classes_gt1 = matfile["classes_r"]            
            classes_gt = []
            for c in classes_gt1[0]:
                if c == "M":
                    classes_gt.append("music")
                if c == "S" or c=="E":
                    classes_gt.append("speech")            
            flagsIndGT, classesAllGT = audioSegmentation.segs2flags([s[0] for s in segs_gt], [s[1] for s in segs_gt], classes_gt, 1.0)            
        if method == "svm" or method == "randomforest" or method == "gradientboosting" or method == "extratrees":
            # speech-music segmentation:            
            [flagsInd, classesAll, acc, CM] = audioSegmentation.mtFileClassification(fileName, modelName, method, False, '')            
        elif method == "hmm":
            [flagsInd, classesAll, _, _] = audioSegmentation.hmmSegmentation(fileName, modelName, PLOT=False, gtFileName="")
        elif method == "cnn":
            WIDTH_SEC = 2.4    
            [Fs, x] = io.readAudioFile(fileName)
            x = io.stereo2mono(x)
            [flagsInd, classesAll, CNNprobs] = mtCNN_classification(x, Fs, WIDTH_SEC, 1.0, RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB)            

        for i in range(flagsIndGT.shape[0]):
            flagsIndGT[i] = classesAll.index(classesAllGT[flagsIndGT[i]])

        #plt.plot(flagsIndGT, 'r')
        #plt.plot(flagsInd)
        #plt.show()

        #print classesAllGT, classesAll
        if postProcess >= 1:
            # medfilt here!            
            flagsInd = scipy.signal.medfilt(flagsInd, 11)
        if postProcess >= 2: #load HMM
            try:
               fo = open(postProcessModelName, "rb")
            except IOError:
               print "didn't find file"
               return
            try:		
               hmm = cPickle.load(fo)
               classesAll = cPickle.load(fo)
            except:
              fo.close()
           

			#Features = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.050*Fs);    # feature extraction
			#[Features, _] = aF.mtFeatureExtraction(x, Fs, mtWin * Fs, mtStep * Fs, round(Fs * 0.050), round(Fs * 0.050))
            flagsInd = hmm.predict(CNNprobs)
            flagsInd = scipy.signal.medfilt(flagsInd, 3)            

        
        if PLOT:
            plt.plot(flagsInd + 0.01)
            plt.plot(flagsIndGT, 'r')
            plt.show()
        CM = np.zeros((2,2))
        for i in range(min(flagsInd.shape[0], flagsIndGT.shape[0])):
            CM[int(flagsIndGT[i]),int(flagsInd[i])] += 1        
        print CM
        return CM, classesAll

def main(argv):    
    '''
    #Segmentation Papameters
    WIDTH_SEC = 2.4    
    RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB = loadCNN()                                    # load the CNN
    [Fs, x] = io.readAudioFile(argv[1])
    x = io.stereo2mono(x)
    mtCNN_classification(x, Fs, WIDTH_SEC, 1.0, RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB)
    '''

    global RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB
    RGB_singleFrame_net, SOUND_mean_RGB, transformer_RGB = loadCNN()                                    # load the CNN

    if argv[1] == "evaluate":
        if os.path.isfile(argv[2]):  
            CM, classesAll = evaluateSpeechMusic(argv[2], argv[3], argv[4], int(argv[5]), argv[6], True)
            print CM
        elif os.path.isdir(argv[2]):    
            CM = np.zeros((2,2))
            types = ('*.wav', )
            wavFilesList = []
            for files in types:
                wavFilesList.extend(glob.glob(os.path.join(argv[2], files)))    
            wavFilesList = sorted(wavFilesList)    
            Recs = []; Pres = []; F1s = [];
            modelName = argv[3]
            method = argv[4]
            postProcess = int(argv[5])                        
            postProcessModelName = argv[6]

            for ifile, wavFile in enumerate(wavFilesList):    
                print "{0:s}, {1:d} file of {2:d}".format(wavFile, ifile+1, len(wavFilesList))
                CMt, classesAll = evaluateSpeechMusic(wavFile, modelName, method, postProcess, postProcessModelName, False)
                CM = CM + CMt                                
                
                [Rec, Pre, F1] = computePreRec(CMt, classesAll)

                print "{0:s}\t{1:s}\t{2:s}\t{3:s}".format("", "Rec", "Pre", "F1")
                for ic, c in enumerate(classesAll):
                    print "{0:s}\t{1:.1f}\t{2:.1f}\t{3:.1f}".format(c, 100*Rec[ic], 100*Pre[ic], 100*F1[ic])

                Recs.append(Rec)
                Pres.append(Pre)
                F1s.append(F1)
            [RecAll, PreAll, F1All] = computePreRec(CM, classesAll)
            Recs = np.mean(np.array(Recs), axis = 0)
            Pres = np.mean(np.array(Pres), axis = 0)
            F1s  = np.mean(np.array(F1s), axis = 0)            
            #print Recs, Pres, F1s
            #print RecAll, PreAll, F1All
            CM = CM / np.sum(CM)
            print CM
            print "Based on overall CM"
            print "{0:s}\t{1:s}\t{2:s}\t{3:s}".format("", "Rec", "Pre", "F1")
            for ic, c in enumerate(classesAll):
                print "{0:s}\t{1:.1f}\t{2:.1f}\t{3:.1f}".format(c, 100*RecAll[ic], 100*PreAll[ic], 100*F1All[ic])                
            
            print "{0:s}\n{1:.1f}\n{2:.1f}\n{3:.1f}\n{4:.1f}\n{5:.1f}".format("forCSV:", 100*RecAll[classesAll.index("speech")], 100*PreAll[classesAll.index("speech")], 100*RecAll[classesAll.index("music")], 100*PreAll[classesAll.index("music")], 100*CM[classesAll.index("speech"),classesAll.index("speech")])
            
            print "Average (duration-irrelevant)"
            print "{0:s}\t{1:s}\t{2:s}\t{3:s}".format("", "Rec", "Pre", "F1")
            for ic, c in enumerate(classesAll):
                print "{0:s}\t{1:.1f}\t{2:.1f}\t{3:.1f}".format(c, 100*Recs[ic], 100*Pres[ic], 100*F1s[ic])
    elif argv[1] == "trainHMM":
        if os.path.isdir(argv[2]):
            modelName = argv[5]
            method = argv[4]
            postProcess = int(argv[6])                 
            hmmModelName = argv[3]       
            trainMetaClassifier(argv[2], hmmModelName, modelName, method, postProcess, False)
    elif argv[1] == "trainHMM_features":
        if os.path.isdir(argv[2]):
            modelName = argv[3]
            trainHMM(argv[2], modelName)




if __name__ == '__main__':
    main(sys.argv)