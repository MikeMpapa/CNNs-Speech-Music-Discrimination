#!/bin/sh
TOOLS=../caffe/build/tools

GLOG_logtostderr=1 $TOOLS/caffe train                       -solver SM_noinit_5000_solver.prototxt
