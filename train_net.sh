#!/bin/sh
#TOOLS=../lisa-caffe-public/build/tools
TOOLS=../caffe/build/tools

GLOG_logtostderr=1 $TOOLS/caffe train                -solver SM_imagenet_2000_noaug_solver.prototxt -weights caffe_imagenet_hyb2_wr_rc_solver_sqrt_iter_310000
