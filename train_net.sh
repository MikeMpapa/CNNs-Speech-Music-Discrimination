#!/bin/sh
TOOLS=../../build/tools

GLOG_logtostderr=1 $TOOLS/caffe train        -solver testWrapper_solver.prototxt -snapshot SOUND_snapshots_singleFrame_RGB_ALL_TRAIN_no_finetune_original_data_iter_3000.solverstate
