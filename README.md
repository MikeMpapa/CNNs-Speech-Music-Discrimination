# CNNs:Speech-Music-Discrimination

##Synopsis
This project describes a new approach to the very traditional problem of Speech-Music Discrimination. According to our knowledge, the proposed method, provides state-of-the-art results on the task. We employ a Deep Convolutional Neural Network (_CNN_) and we offer a compact framework to perform segmentation and binary (Speech/Music) classification. Our method is unchained from traditional audio features, which offer inferior results on the task as shown in (--reference to the paper--). Instead it exploits the highly invariant features produced by CNNs and opperates on pseudocolored RGB or grayscale frequency-images, which represent audio segments. 

**The repository consists of the following modules:**
 * Audio segmentation using the [PyAudio](https://github.com/tyiannak/pyAudioAnalysis.git) analysis lybrary
 * CNN training using the [CAFFE Deep-Learning Framework](https://github.com/BVLC/caffe).  
 * Audio classification using: 
  * CNNs
  * CNNs + median_filtering 
  * CNNs + median_filtering + HMMs
 * Two pretrained CNNs on the task of Speech/Music Discrimination. The network can be also used for weight initialization for other similar tasks. 
 * An audio dataset consisting of more than 10h continous audio streams. At this point the data are available in the form of spectrograms.

##Installation
- Dependencies
 1. [PyAudio](https://github.com/tyiannak/pyAudioAnalysis.git) 
 2. [CAFFE Deep-Learning Framework](http://caffe.berkeleyvision.org/installation.html)

 _* Installation instructions offered in detail on the above links_

- Add Caffe to your working dir
 1. [trainCNN.py](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/trainCNN.py) --> Line:4
 2. [train_net.sh](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/train_net.sh) --> Line:2
 3. [ClassifyWav.py](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/ClassifyWav.py) --> Line:14
 
 **or add pycaffe to your .bashrc for directory independent access**
 
 * open .bashrc file located at your home directory 
   In a terminal type:
    1. ```cd ~ ``` to navigate to your home directory
    2.   ```ls -a ``` to see the file listed
    3. ```nano .bashrc ``` to open the file in terminal
    4. scroll at the botom of the file and add: 
    
       export PYTHONPATH=$PYTHONPATH:"/home/--myPathToCaffe--/caffe/python"
       
       , where _--myPathToCaffe--_ is the path to the caffe library as it appears in your local machine
       
       i.e.: export PYTHONPATH=$PYTHONPATH:"/home/michalis/Liraries/caffe/python"
    5.  ```source ~/.bashrc``` to update your source file    


##Code Description

#### **Data Preparation**
   1. Convert your audio files into pseudocolored RGB or grayscale spectrogram images using _generateSpectrograms.py_
      _TO BE UPDATED a)How to run, b)How to set segmentation parameters c) How the output looks like_

   2. Split the spectrogram images into train and test as shown in Fig1:
 
   <img src="https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/train-test.PNG" width="500" height="300">
   <figcaption>Fig1. - Data Structure</figcaption>

    * Train/Test and Classes represent directories 
    
    * Samples represent files
     
    * If you wish to use the architecture proposed in this work:
    
      1. Data should be pseudo-colored RGB spectrogram images of size 227x227 as shown in Fig2
    <img src="https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/sampleIMg.png" width="227" height="227">
    <figcaption>Fig2. - Sample RGB Spectrogram</figcaption>
    
      2. or grayscale spectrogram images of size 200x200 as shown in Fig3
      
         <img src="https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/sampleIMg2.png" width="220" height="220">
    <figcaption>Fig3. - Sample Grayscale Spectrogram</figcaption>
    
      * Image resizing can be done directly using CAFFE framework.
  
#### **Training** 

  * **Train a CNN**
  
  1. Provide Network Architecture file. You can use one of the proposed architectures ([_SpeechMusic\_RGB.prototxt_](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/SpeechMusic_RGB.prototxt), [_SpeechMusic\_GRAY.prototxt_](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/SpeechMusic_Gray.prototxt) ) or another CNN of your choice.

2. Train
 
    Training can be done either by training a new network from sratch or by finetuning a pretrained architecture. 
    
    The pretrained model used in the paper for fine-tuning is the caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000 initially proposed in [Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015](http://arxiv.org/abs/1411.4389). To exploit the weight initialization of the pretrained model use the CNN architecture shown in [_SpeechMusic\_RGB.prototxt_](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/SpeechMusic_RGB.prototxt). 
    
    If you wish to deploy the smaller CNN architecture that operates on grayscale images you should use the CNN architecture shown in [_SpeechMusic\_GRAY.prototxt_](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/SpeechMusic_Gray.prototxt). This model was trained from scratch without weight initialization. 

    * Train from scratch:
   ```shell
python trainCNN.py <architecture_file>.prototxt <path_to_train_data_root_foler> <path_to_test_data_root_foler> <snapshot_prefix> <total_number_of_iterations>
``` 
    * Finetune pretrained network:
   ```shell
python trainCNN.py <architecture_file>.prototxt <path_to_train_data_root_foler> <path_to_test_data_root_foler> <snapshot_prefix> <total_number_of_iterations> --init <pretrained_network>.caffemodel --init_type fin
``` 
    * Resume Training:
   ```shell
python trainCNN.py <architecture_file>.prototxt <path_to_train_data_root_foler> <path_to_test_data_root_foler> <snapshot_prefix> <total_number_of_iterations> --init <pretrained_network>.solverstate --init_type res
``` 
    * For more details about modifying other learning parameters (i.e learning rate, step size etc.) type:
    ```shell 
    python trainCNN.py -h
    ``` 
  3. Outputs:
     1. _\<snapshot_prefix\>_solver.prototxt_
         Solver file required by caffe to train the CNN. The solver file describes all the parameters of the current experients. Commented lines have additional information regarding the experiments that are not required by the Caffe framework.
     2. _\<snapshot_prefix\>_TrainSource.txt_ & _\<snapshot_prefix\>_TestSource.txt_
         Full paths to training and test samples with each samples class
     
  * **Train HMM**
  
     **TO BE UPDATED**

####  **Classification**

##Code Example
 * Generate Spectrogram Images:
 * Train from scratch:
 
   ```shell
python trainCNN.py SpeechMusic_RGB.prototxt Train Test myOutput 4000
```
 * Finetune pretrained network (_train and test paths are according to Fig1_):
 
   ```shell
python trainCNN.py SpeechMusic_RGB.prototxt Train Test myOutput 1000 --init caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000.caffemodel --init_type fin
``` 
 * Resume training from pretrained network (_train and test paths are according to Fig1_):
 
   ```shell
python trainCNN.py SpeechMusic_RGB.prototxt Train Test my_new_Output 2000 --init myOutput.solverstate --init_type res
``` 

##Coclusions
We provide a new method for the task of Speech/Music Discrimination using Convolutional Neural Networks.
The main contributions of this work are the following:

1. A compact framework for:
       * Segmenting and Classifying long audio streams into Speech and Music segments.
       * Train new CNN models on binary audio tasks
2. A big dataset on long audio streams (more than 10h) for the task of speech music discrimination. The dataset is provided in the form of spectrograms.

3. Two different pretrained CNN architectures that can be used for weight initialization for other binary classification tasks. 

##References & Citations
Please use the following citations if you experimented with _CNNs:Speech-Music-Discrimination_ project:

**CNNs:Speech-Music-Discrimination**
pending....

**PyAudioAnalysis**
@article{giannakopoulos2015pyaudioanalysis,
  title={pyAudioAnalysis: An Open-Source Python Library for Audio Signal Analysis},
  author={Giannakopoulos, Theodoros},
  journal={PloS one},
  volume={10},
  number={12},
  year={2015},
  publisher={Public Library of Science}
}


**Caffe Framework**
@article{jia2014caffe,
  Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
  Journal = {arXiv preprint arXiv:1408.5093},
  Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
  Year = {2014}
}

If you used the pretrained network **_caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000_**  for your experiments, please also cite:

@inproceedings{donahue2015long,
  title={Long-term recurrent convolutional networks for visual recognition and description},
  author={Donahue, Jeffrey and Anne Hendricks, Lisa and Guadarrama, Sergio and Rohrbach, Marcus and Venugopalan, Subhashini and Saenko, Kate and Darrell, Trevor},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2625--2634},
  year={2015}
}
