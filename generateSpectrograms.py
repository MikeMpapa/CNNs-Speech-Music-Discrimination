#!/usr/bin/env python2.7
import scipy.misc
import argparse
import os
import sys
import audioop
import numpy
import glob
import scipy
import subprocess
import wave
import cPickle
import threading
import shutil
import ntpath
import random
import matplotlib.pyplot as plt
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioVisualization as aV
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import utilities as uT
import scipy.io.wavfile as wavfile
import matplotlib.patches
import Image
import cv2
import matplotlib.cm


def createSpectrogramFile(x, Fs, fileName, stWin, stStep):
        specgramOr, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs * stWin), round(Fs * stStep), False)            
        print specgramOr.shape
        if inputs[2]=='full':
        	print specgramOr
        	numpy.save(fileName.replace('.png','')+'_spectrogram', specgramOr)
        else:	
	        #specgram = scipy.misc.imresize(specgramOr, float(227.0) / float(specgramOr.shape[0]), interp='bilinear')                        
	        specgram = cv2.resize(specgramOr,(227, 227), interpolation = cv2.INTER_LINEAR)
	        im1 = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))
	        scipy.misc.imsave(fileName, im1)

def main(argv):
	if argv[2]=='full':
	    dirName = argv[1]
	    types = ('*.wav', )
	    filesList = []
	    for files in types:
	        filesList.extend(glob.glob(os.path.join(dirName, files)))
	    filesList = sorted(filesList)
	    
	    filesListIrr = []
	    
	    filesListIrr = sorted(filesListIrr)

	    stWin = 0.020
	    stStep = 0.015
	    for f in filesList:
	        [Fs, x] = audioBasicIO.readAudioFile(f)
	        x = audioBasicIO.stereo2mono(x)
	        createSpectrogramFile(x, Fs, f.replace(".wav",".png"), stWin, stStep)

	else:
	    dirName = argv[1]
	    dirNameIrrelevant = argv[2]
	    types = ('*.wav', )
	    filesList = []
	    for files in types:
	        filesList.extend(glob.glob(os.path.join(dirName, files)))
	    filesList = sorted(filesList)
	    
	    filesListIrr = []
	    for files in types:
	        filesListIrr.extend(glob.glob(os.path.join(dirNameIrrelevant, files)))
	    filesListIrr = sorted(filesListIrr)
	    print filesListIrr

	    WIDTH_SEC = 1.5
	    stWin = 0.040
	    stStep = 0.005
	    WIDTH = WIDTH_SEC / stStep

	    for f in filesList:
	    	print f
	        [Fs, x] = audioBasicIO.readAudioFile(f)
	        x = audioBasicIO.stereo2mono(x)        
	        x = x.astype(float) / x.max()        
	        for i in range(3):
	            if x.shape[0] > WIDTH_SEC * Fs + 200:
	                randStartSignal = random.randrange(0, int(x.shape[0] - WIDTH_SEC * Fs - 200) )
	                x2 = x[randStartSignal : randStartSignal + int ( (WIDTH_SEC + stStep) * Fs) ]
	                createSpectrogramFile(x2, Fs, f.replace(".wav",".png"), stWin, stStep)														# ORIGINAL

	                if len(dirNameIrrelevant) > 0:
		                # AUGMENTED
		                randIrrelevant = random.randrange(0, len(filesListIrr))
		                [Fs, xnoise] = audioBasicIO.readAudioFile(filesListIrr[randIrrelevant])
		                xnoise = xnoise.astype(float) / xnoise.max()            
		                
		                randStartNoise = random.randrange(0, xnoise.shape[0] - WIDTH_SEC * Fs - 200)
		                R = 5; xN = (R * x2.astype(float)  + xnoise[randStartNoise : randStartNoise + x2.shape[0]].astype(float)) / float(R+1)
		                wavfile.write(f.replace(".wav","_rnoise{0:d}1.wav".format(i)), Fs, (16000 * xN).astype('int16'))
		                createSpectrogramFile(xN, Fs, f.replace(".wav","_rnoise{0:d}1.png".format(i)), stWin, stStep)

		                randStartNoise = random.randrange(0, xnoise.shape[0] - WIDTH_SEC * Fs - 200)
		                R = 4; xN = (R * x2.astype(float)  + xnoise[randStartNoise : randStartNoise + x2.shape[0]].astype(float)) / float(R+1)
		                wavfile.write(f.replace(".wav","_rnoise{0:d}2.wav".format(i)), Fs, (16000 * xN).astype('int16'))
		                createSpectrogramFile(xN, Fs, f.replace(".wav","_rnoise{0:d}2.png".format(i)), stWin, stStep)

		                randStartNoise = random.randrange(0, xnoise.shape[0] - WIDTH_SEC * Fs - 200)
		                R = 3; xN = (R * x2.astype(float)  + xnoise[randStartNoise : randStartNoise + x2.shape[0]].astype(float)) / float(R+1)
		                wavfile.write(f.replace(".wav","_rnoise{0:d}3.wav".format(i)), Fs, (16000 * xN).astype('int16'))
		                createSpectrogramFile(xN, Fs, f.replace(".wav","_rnoise{0:d}3.png".format(i)), stWin, stStep)

		                #specgramOr, TimeAxis, FreqAxis = aF.stSpectogram(x2, Fs, round(Fs * stWin), round(Fs * stStep), False)
		                #im2 = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))
		                #plt.subplot(2,1,1)
		                #plt.imshow(im1)
		                #plt.subplot(2,1,2)
		                #plt.imshow(im2)
		                #plt.show()

		                '''
		                if int(specgramOr.shape[0]/2) - WIDTH/2 - int((0.2) / stStep) > 0:
		                    specgram = specgramOr[int(specgramOr.shape[0]/2) - WIDTH/2 - int((0.2) / stStep):int(specgramOr.shape[0]/2) + WIDTH/2 - int((0.2) / stStep), :]

		                    specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')
		                    im = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))                
		                    print specgram.shape
		                    scipy.misc.imsave(f.replace(".wav","_02A.png"), im)

		                    specgram = specgramOr[int(specgramOr.shape[0]/2) - WIDTH/2 + int((0.2) / stStep):int(specgramOr.shape[0]/2) + WIDTH/2 + int((0.2) / stStep), :]                
		                    specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')                
		                    print specgram.shape
		                    im = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))                
		                    scipy.misc.imsave(f.replace(".wav","_02B.png"), im)

		                    # ONLY FOR SPEECH (fewer samples). Must comment for music
		                    specgram = specgramOr[int(specgramOr.shape[0]/2) - WIDTH/2 - int((0.1) / stStep):int(specgramOr.shape[0]/2) + WIDTH/2 - int((0.1) / stStep), :]                
		                    specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')                        
		                    im = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))                
		                    print specgram.shape
		                    scipy.misc.imsave(f.replace(".wav","_01A.png"), im)

		                    specgram = specgramOr[int(specgramOr.shape[0]/2) - WIDTH/2 + int((0.1) / stStep):int(specgramOr.shape[0]/2) + WIDTH/2 + int((0.1) / stStep), :]                
		                    specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')                
		                    print specgram.shape
		                    im = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))                
		                    scipy.misc.imsave(f.replace(".wav","_01B.png"), im)


		                    if int(specgramOr.shape[0]/2) - WIDTH/2 - int((0.5) / stStep) > 0:
		                        specgram = specgramOr[int(specgramOr.shape[0]/2) - WIDTH/2 - int((0.5) / stStep):int(specgramOr.shape[0]/2) + WIDTH/2 - int((0.5) / stStep), :]                
		                        specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')                        
		                        im = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))                
		                        print specgram.shape
		                        scipy.misc.imsave(f.replace(".wav","_02A.png"), im)

		                        specgram = specgramOr[int(specgramOr.shape[0]/2) - WIDTH/2 + int((0.5) / stStep):int(specgramOr.shape[0]/2) + WIDTH/2 + int((0.5) / stStep), :]                
		                        specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')                
		                        print specgram.shape
		                        im = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))                
		                        scipy.misc.imsave(f.replace(".wav","_02B.png"), im)
		                '''
	




if __name__ == '__main__':
    inputs = sys.argv
    global inputs
    main(inputs)
