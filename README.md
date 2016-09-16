# CNNs:Speech-Music-Discrimination

##Synopsis
This project describes a new approach to the very traditional problem of Speech-Music Discrimination. According to our knowledge, the proposed method, provides state-of-the-art results on the task. We employ a Deep Convolutional Neural Network (_CNN_) and we offer a compact framework to perform binary (Speech/Music) classification both in short audio segments and also in whole .wav streams. Our method is unchained from traditional audio features, which offer inferior results on the task as shown in (--reference to the paper--). Instead it exploits the highly invariant features produced by CNNs and opperates on pseudocolored RGB frequency-images, which represent wav segments. 

**We offer a mechanism for:**
 * Audio segmentation using the [PyAudio](https://github.com/tyiannak/pyAudioAnalysis.git) analysis lybrary
 * CNN training using the [CAFFE Deep-Learning Framework](https://github.com/BVLC/caffe)
 * Audio classification using: 
  1. CNNs
  2. CNNs + median_filtering 
  3. CNNs + median_filtering + HMMs

##Motivation

##Installation

##Dependencies

##Code Description

##Code Example
--how to run and outputs

##Contributors

##References


Cite
