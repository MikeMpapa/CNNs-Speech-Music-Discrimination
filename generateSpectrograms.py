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
import matplotlib.cm

def main(argv):
    dirName = argv[1]
    types = ('*.wav', )
    filesList = []
    for files in types:
        filesList.extend(glob.glob(os.path.join(dirName, files)))
    filesList = sorted(filesList)
    WIDTH_SEC = 2.4
    stWin = 0.020
    stStep = 0.015
    WIDTH = WIDTH_SEC / stStep

    for f in filesList:
        [Fs, x] = audioBasicIO.readAudioFile(f)
        x = audioBasicIO.stereo2mono(x)
        specgramOr, TimeAxis, FreqAxis = aF.stSpectogram(x, Fs, round(Fs * stWin), round(Fs * stStep), False)
        if specgramOr.shape[0]>WIDTH:
            specgram = specgramOr[int(specgramOr.shape[0]/2) - WIDTH/2:int(specgramOr.shape[0]/2) + WIDTH/2, :]            
            specgram = scipy.misc.imresize(specgram, float(227.0) / float(specgram.shape[0]), interp='bilinear')
            print specgram.shape            
            im = Image.fromarray(numpy.uint8(matplotlib.cm.jet(specgram)*255))
            #plt.imshow(im)
            scipy.misc.imsave(f.replace(".wav",".png"), im)

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


if __name__ == '__main__':
    main(sys.argv)
