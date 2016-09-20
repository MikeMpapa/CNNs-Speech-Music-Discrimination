# CNNs:Speech-Music-Discrimination

##Synopsis
This project describes a new approach to the very traditional problem of Speech-Music Discrimination. According to our knowledge, the proposed method, provides state-of-the-art results on the task. We employ a Deep Convolutional Neural Network (_CNN_) and we offer a compact framework to perform segmentation and binary (Speech/Music) classification. Our method is unchained from traditional audio features, which offer inferior results on the task as shown in (--reference to the paper--). Instead it exploits the highly invariant features produced by CNNs and opperates on pseudocolored RGB frequency-images, which represent wav segments. 

**The repository consists of the following modules:**
 * Audio segmentation using the [PyAudio](https://github.com/tyiannak/pyAudioAnalysis.git) analysis lybrary
 * CNN training using the [CAFFE Deep-Learning Framework](https://github.com/BVLC/caffe).  
 * Audio classification using: 
  * CNNs
  * CNNs + median_filtering 
  * CNNs + median_filtering + HMMs
 * A pretrained CNN for the task of Speech/Music Discrimination. The network can be also used for weight initialization for other similar tasks. 

##Installation
- Dependencies
 1. [PyAudio](https://github.com/tyiannak/pyAudioAnalysis.git) 
 2. [CAFFE Deep-Learning Framework](http://caffe.berkeleyvision.org/installation.html)
 
_* Installation instructions offered in detail on the above links_

##Code Description

#### **Data Preparation**
   1. Convert your wav files into pseudocolored RGB spectrogram images using _generateSpectrograms.py_
      _TO BE UPDATED a)How to run, b)How to set segmentation parameters c) HOw the output looks like_

   2. Split the spectrogram images into train and test as shown in Fig1:
 
   <img src="https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/train-test.PNG" width="500" height="300">
   <figcaption>Fig1. - Data Structure</figcaption>

    * Train/Test and Classes represent directories whereas Samples represent files
     
    * Data should be pseudo-colored RGB spectrogram images of size 227x227 as shown in Fig2
    <img src="https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/sampleIMg.png" width="227" height="227">
    <figcaption>Fig2. - Sample Spectrogram</figcaption>
  
#### **Training** 

  * Train a CNN
  
  1. Provide Network Architecture file ([_SpeechMusic\_RGB.prototxt_](https://github.com/MikeMpapa/CNNs-Speech-Music-Discrimination/blob/master/SpeechMusic_RGB.prototxt) ).
     You can use any other CNN architecture of your choice. This one is the proposed architecture as described in the paper. 
  2. Train
 
    Training can be done either by training a new network from sratch or by finetuning a pretrained architecture. The pretrained model used in the paper for fine-tuning is the caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000 initially proposed in [Donahue, Jeffrey, et al. "Long-term recurrent convolutional networks for visual recognition and description." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2015.](http://arxiv.org/abs/1411.4389)

    * Train from scratch:
   ```shell
python trainCNN.py <architecture_file>.prototxt <path_to_train_data_root_foler> <path_to_test_data_root_foler> <snapshot_prefix> <total_number_of_iterations>
``` 
    * Finetune pretrained network:
   ```shell
python trainCNN.py <architecture_file>.prototxt <path_to_train_data_root_foler> <path_to_test_data_root_foler> <snapshot_prefix> <total_number_of_iterations> --init <pretrained_network>.caffemodel --init_type fin
``` 
    * For more details about modifying other learning parameters (i.e learning rate, step size etc.) type:
    ```shell 
    python trainCNN.py -h
    ``` 
    * Outputs:
     1. \<snapshot_prefix\>_solver.prototxt
         Solver file required by caffe to train the CNN. The solver file describes all the parameters of the current experients. Commented lines have additional information regarding the experiments that are not required by the Caffe framework.
     2. \<snapshot_prefix\>_TrainSource.txt & \<snapshot_prefix\>_TestSource.txt
         Full paths to training and test samples with each samples class
     
  * Train HMM

####  **Classification**

##Code Example
 * Generate Spectrogram Images:
 * Train from scratch:
 
   ```shell
python trainCNN.py SpeechMusic_RGB.prototxt Train Test myOutput 4000
```
 * Finetune pretrained network (train and test paths are according to Fig1_):
 
   ```shell
python trainCNN.py SpeechMusic_RGB.prototxt Train Test myOutput 1000 --init caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000.caffemodel --init_type fin
``` 

##Contributors

##References


##Cite
