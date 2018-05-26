#!/usr/bin/env python
"""Chainer example: train a classifier 
"""
import argparse
import os
import json
import sys

import numpy as np
from PIL import Image
import random

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer.dataset import dataset_mixin
from chainer import training
from chainer.training import extensions

import chainer.links as L
import chainer.functions as F
from chainer.dataset import concat_examples

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import train_classifier as my_train


def main():
    parser = argparse.ArgumentParser(description='Predict classes with the trained classifier')
    parser.add_argument('DATA_DIR', type=str,
                        help='It includes test.txt.')
    parser.add_argument('MODEL', type=str, 
                        help='The trained model')

    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='size of data')
    
    args = parser.parse_args()
    batchsize = args.batchsize
    crop_size = args.crop_size
    test_txt = os.path.join(args.DATA_DIR, 'test.txt')
    mean_file = os.path.join(args.DATA_DIR, 'mean.npy')

    
    model = my_train.resnet50(2) # 引数はクラス数
    print('Load model from', args.MODEL)
    serializers.load_npz(args.MODEL, model)
    
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()


    # Prepare dataset
    print('load my dataset')
    mean = np.load(mean_file)
    test = my_train.PreprocessedDataset(test_txt, mean, crop_size, random=False)
    test_iter  = chainer.iterators.SerialIterator(test, batchsize,
                                                  repeat=False, shuffle=True)

    print('test: {}'.format(test.__len__()))

    y = []
    t = []
    for i, batch in enumerate(test_iter):
        print(i+1)
        batch = concat_examples(batch, device=args.gpu)
        with chainer.using_config('train', False), \
             chainer.using_config('enable_backprop', False):
            out = model.predict(batch[0])
        ans = batch[1]
        y.append(out)
        t.append(ans)
        
    y = F.concat(y, axis=0)
    t = F.concat(t, axis=0)
    y = y.reshape((len(y), )).data
    t = t.reshape((len(t), )).data

    # to gpu -> cpu
    if args.gpu >= 0:
        y = chainer.cuda.to_cpu(y)
        t = chainer.cuda.to_cpu(t)
    
    precision = precision_score(t, y, pos_label=0)
    recall = recall_score(t, y, pos_label=0)
    f1 = f1_score(t, y, pos_label=0)
    print('{}, {}, {}'.format(precision, recall, f1))
    

    
           
if __name__ == '__main__':
    main()
