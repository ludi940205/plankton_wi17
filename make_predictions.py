'''
/*******************************************************************************
 * make_predictions.py
 *
 * Using this script, we will execute predictions on new unseen data from our
 * our trained model
 *
 * Script Architecture:
 * 1) Read Mean Image
 * 2) Read model architecture and trained weights
 * 3) Define image transformers
 * 4) Apply image processing steps to training phase
 * 5) Calculate each class's probability
 * 6) Print class with each probability
 *
*******************************************************************************/
'''
import numpy as np
import cv2
import os, glob, random, caffe, lmdb
from caffe.proto import caffe_pb2

#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/data4/plankton_wi17/mnist_data/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

#Read model architecture and trained weights
net = caffe.Net('/data4/plankton_wi17/mnist_data/caffenet_deploy.prototxt',
                '/data4/plankton_wi17/mnist_data/caffe_model_iter_10000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))