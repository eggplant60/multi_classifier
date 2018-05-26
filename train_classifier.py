#!/usr/bin/env python
"""Chainer example: train a classifier 
"""
#from __future__ import print_function
import argparse
import os
import json
import sys

import numpy as np
from PIL import Image
import random
import pandas as pd

import chainer
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer.dataset import dataset_mixin
from chainer import training
from chainer.training import extensions
import cupy

import chainer.links as L
import chainer.functions as F

import create_dataset

'''
LabeledImageDataset だと以下の点が不都合
1. RGBA に対応していない
2. カンマ区切りのファイルに対応していない
'''
class MyLabeledImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, pairs):
        if isinstance(pairs, str):
            pairs_path = pairs
            with open(pairs_path) as pairs_file:
                pairs = []
                for i, line in enumerate(pairs_file):
                    pair = line.strip().split(',')
                    if len(pair) != 2:
                        raise ValueError(
                            'invalid format at line {} in file {}'.format(
                                i, pairs_path))
                    pairs.append((pair[0], pair[1]))
        self.pairs = pairs
              
    def __len__(self):
        return len(self.pairs)

    def get_example(self, i):
        path, str_label = self.pairs[i]
        f = Image.open(path)
        #f = f.convert('RGB') # for greyscale or RGBA
        image = np.asarray(f, dtype=np.float32)
        # 基本的にはリサイズ時にカラー画像に変更しているが、
        # まれにgreyscaleのままの場合があるため
        if image.ndim == 2:
            # image is greyscale
            image = image[:, :, np.newaxis]
        image = image.transpose(2, 0, 1)
        onehot_label = create_dataset.str_to_onehot(str_label)
        label = np.array(onehot_label, dtype=np.int32)
        return image, label


'''
chainerに用意されている前処理用のクラスを継承
'''
class PreprocessedDataset(dataset_mixin.DatasetMixin):

    def __init__(self, path, mean, crop_size, random=True):
        self.base = MyLabeledImageDataset(path)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)
    
    # 学習中に逐次実行される
    def get_example(self, i):
        image, label = self.base[i]
        
        _, h, w = image.shape
        crop_size = self.crop_size
        
        # Cropping (random or center rectangular) and random horizontal flipping
        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image = image - self.mean[:, top:bottom, left:right] # -= だとブロードキャストできない？
        image *= (1.0 / 255.0)  # Scale to [0, 1]

        return image, label
    
'''
return: recall, precision
'''
def evaluate_accuracy(h, t, suggestion=10):
    index_prod = cupy.argsort(h.data, axis=1)[:, ::-1][:, :suggestion]  # h: Variable
    y_prod = cupy.zeros(h.shape, dtype=np.int32)
    for i, index_one_batch in enumerate(index_prod):
        y_prod[i, index_one_batch] = 1
    #print(y_prod)
    y_true = t                   # t: ndarray

    valid = cupy.any(y_true, axis=1)
    y_and = cupy.logical_and(y_prod, y_true)
    num = cupy.sum(y_and, axis=1)
    den = cupy.sum(y_true, axis=1)
    #print(num)
    #print(den)
    return cupy.mean(num[valid]/den[valid]), \
        cupy.mean(num/suggestion)
        


'''
Resnet50
'''
class resnet50(chainer.Chain):
    def __init__(self, class_size, loss_weight):
        super(resnet50, self).__init__()
        with self.init_scope():
            self.model = L.ResNet50Layers(pretrained_model='auto')
            self.fc6 = L.Linear(2048, class_size)
        #print(self.model.available_layers)
        self.loss_weight = cuda.to_gpu(loss_weight)
        self.class_size  = class_size
        
    def __call__(self, x, t):
        batchsize = x.shape[0]
        h = self.model(x, layers=['pool5'])['pool5']
        h = self.fc6(h)
        score = F.sigmoid_cross_entropy(h, t, reduce='no')
        loss = F.average(score * self.loss_weight)
        #loss = F.softmax_cross_entropy(h, t)
        
        # Note: binary_accuracy はマルチクラス分類に適切ではない
        #chainer.report({'loss': loss, 'accuracy': F.binary_accuracy(h, t)}, self)
        recall, precision = evaluate_accuracy(h, t)
        chainer.report({'loss': loss, 'recall': recall, 'precision': precision}, self)
        return loss

    # def predict(self, x):
    #     h = self.model(x, layers=['pool5'])['pool5']
    #     h = self.fc6(h)
    #     y = F.argmax(h, axis=1)
    #     return y


def main():
    parser = argparse.ArgumentParser(description='Train a r18 classifier with pixiv dataset')
    parser.add_argument('DATA_DIR', type=str,
                        help='It includes train.txt, val.txt, and mean.npy.')
    parser.add_argument('--initmodel', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', default='',
                        help='Resume the optimization from snapshot')

    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--n_epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='learning minibatch size')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='size of data')
    # parser.add_argument('--mergin', defult=0.5, type=int,
    #                     help='mergin size for random crop')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='number of iteration to show log')
    parser.add_argument('--snap-interval', type=int, default=10,
                        help='number of iteration to show log')
    parser.add_argument('--out', '-o', type=str, default='result',
                        help='Directory to output the result')
    
    args = parser.parse_args()
    n_epoch = args.n_epoch
    batchsize = args.batchsize
    crop_size = args.crop_size
    train_txt = os.path.join(args.DATA_DIR, 'train.txt')
    val_txt   = os.path.join(args.DATA_DIR, 'val.txt')
    vocab_file = os.path.join(args.DATA_DIR, 'vocab.txt')
    mean_file = os.path.join(args.DATA_DIR, 'mean.npy')

    # 学習条件をファイルに書き出す
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    with open(os.path.join(args.out, 'args.txt'), 'w') as f:
        args_dict = {}
        for i in dir(args):
            if '_' in i[0]: continue
            args_dict[str(i)] = getattr(args, i)
        json.dump(args_dict, f, ensure_ascii=False,
                  indent=4, sort_keys=True, separators=(',', ': '))
    
    print('# GPU: {}'.format(args.gpu))
    print('# n_epoch: {}'.format(n_epoch))
    print('# Minibatch-size: {}'.format(batchsize))
    print('')

    # read vocaburary
    vocab = pd.read_csv(vocab_file, header=None, names=['id', 'tag', 'appear'])
    class_size = len(vocab)
    a = vocab['appear'].values.astype('f')
    a = np.reciprocal(a)
    loss_weight = a / sum(a)
    
    # model
    model = resnet50(class_size, loss_weight)
    if args.initmodel:
        print('Load model from', args.initmodel)
        serializers.load_npz(args.initmodel, model)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    #xp = np if args.gpu < 0 else cuda.cupy

    
    # Prepare dataset
    print('load my dataset')
    mean  = np.load(mean_file)
    train = PreprocessedDataset(train_txt, mean, crop_size)
    val   = PreprocessedDataset(val_txt,  mean, crop_size)
    print('train: {}, val: {}'.format(train.__len__(), val.__len__()))
    train_iter = chainer.iterators.MultiprocessIterator(train, batchsize) # デフォルトで全core使用
    val_iter  = chainer.iterators.MultiprocessIterator(val, batchsize,
                                                        repeat=False, shuffle=False)
    
    # Setup optimizer
    #optimizer = optimizers.Adam() # worse for CNN?
    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    
    trainer = training.Trainer(updater, (n_epoch, 'epoch'), out=args.out)

    log_interval = (args.log_interval, 'iteration')
    val_interval = (args.log_interval, 'iteration')
    snap_interval = (args.snap_interval, 'epoch')
        
    trainer.extend(extensions.Evaluator(val_iter, model, device=args.gpu),
                   trigger=val_interval)
                                        
    trainer.extend(extensions.dump_graph('main/loss'))
    
    trainer.extend(extensions.snapshot(), trigger=snap_interval)
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'), trigger=snap_interval)
           
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.observe_lr(), trigger=log_interval)
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'validation/main/loss',
        'main/recall', 'validation/main/recall',
        'main/precision', 'validation/main/precision',
        'lr'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar())

    # Resume
    if args.resume:
        print('Load optimizer state from', args.resume)
        serializers.load_npz(args.resume, optimizer)

    print('start training')
    trainer.run()

    
if __name__ == '__main__':
    main()
