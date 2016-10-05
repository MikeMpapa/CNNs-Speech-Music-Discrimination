#!/bin/sh
TOOLS=../caffe/build/tools

GLOG_logtostderr=1 $TOOLS/caffe train                     -solver SM_imagenet_10000_aug_solver.prototxt -weights caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000
