#!/usr/bin/env python
# -*- coding: utf-8 -*-

#import time
import argparse
from pymongo import MongoClient
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random
import csv
import sys
from multiprocessing import Pool
import re
from collections import Counter
import pandas as pd

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # for large files

import train_classifier
import compute_mean

def onehot_to_str(onehot):
    return ''.join(map(str, onehot))

def str_to_onehot(string):
    return [int(s) for s in string]



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read DB, resize pictures, and create a dataset')
    parser.add_argument('DATA_SIZE', type=int)
    parser.add_argument('NUM_CLASS', type=int)
    
    parser.add_argument('--val',  type=float, default=0.1)
    parser.add_argument('--test', type=float, default=0.1)
    
    parser.add_argument('--out', '-o',  type=str, default='./dataset')
    parser.add_argument('--train_txt', type=str, default='train.txt')
    parser.add_argument('--val_txt', type=str, default='val.txt')
    parser.add_argument('--test_txt', type=str, default='test.txt')
    parser.add_argument('--vocab_txt', type=str, default='vocab.txt')
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resize', type=int, default=256)

    args = parser.parse_args()
    train_txt = os.path.join(args.out, args.train_txt)
    val_txt = os.path.join(args.out, args.val_txt)
    test_txt = os.path.join(args.out, args.test_txt)
    vocab_txt = os.path.join(args.out, args.vocab_txt)
                             
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    random.seed(args.seed)

    
    '''1. Read DB'''
    client = MongoClient('localhost', 27017)
    collection = client.scraping.pixiv


    db_size = collection.find().count()
    print('Number of entries on DB: {}'.format(db_size))
    assert db_size >= args.DATA_SIZE 

    # gif は RGBA の変換でエラーすることがある
    query = {'local_path': re.compile(r'.+\.(?!gif)')}
    projection = {'_id': False, 'local_path': True, 'tags': True}
    df_pairs = pd.DataFrame(list(collection.find(query, projection)))
    df_pairs = df_pairs.sample(args.DATA_SIZE)

    paths = df_pairs['local_path'].values.tolist()
    tags = df_pairs['tags'].values.tolist()
    
    
    '''2. Make vocaburary and labels'''
    tags_flat = [t for inner_list in tags for t in inner_list]
    # users入りは他のタグと重複するので削除
    tags_flat = list(filter(lambda t: re.search('.*users入り', t)==None, tags_flat))
    # 件数の多い順に NUM_CLASS 個のタグを抽出
    common_tags = Counter(tags_flat).most_common(args.NUM_CLASS)

    # ID, tag_name, num of appear
    vocab_mat = [[i, t[0], t[1]] for i, t in enumerate(common_tags)]    
    with open(vocab_txt, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(vocab_mat)
    
    classes = [t[1] for t in vocab_mat]

    labels = []
    for tag in tags:
        label = [1 if (c in tag) else 0 for c in classes]
        label = onehot_to_str(label)
        labels.append(label)

         
          
    '''3. Resize pictures'''
    def resize_pictures(paths, dir_name, stdout): # stdout: if progress is printed
        new_dir = os.path.join(args.out, dir_name)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        new_paths = []
        N = len(paths)
        for i, path in enumerate(paths):
            new_path = os.path.join(new_dir, os.path.basename(path))
            new_paths.append(new_path)
            if not os.path.exists(new_path):
                img = Image.open(path)
                img = img.convert('RGB') # for greyscale or RGBA
                img = img.resize((args.resize, args.resize), Image.ANTIALIAS)
                img.save(new_path)
            if stdout:
                sys.stderr.write('{} / {}\r'.format(i+1, N))
                sys.stderr.flush()
        if stdout:
            sys.stderr.write('\n')
        return new_paths

    def wrapper(args):
        return resize_pictures(*args)

    print('resizing pictures...')
    p = Pool(2)
    paths1 = paths[:len(paths)//2] # split data
    paths2 = paths[len(paths)//2:] # split data
    w_pairs = [(paths1, 'pictures', False), (paths2, 'pictures', True)]
    new_paths1, new_paths2 = p.map(wrapper, w_pairs)
    p.close()

    paths = new_paths1 + new_paths2

    
    '''4. Create dataset'''
    def split_train_val_test(paths, labels, val_ratio, test_ratio):
        n_all  = len(paths)
        n_val  = int(n_all * val_ratio)
        n_test = int(n_all * test_ratio)
        train_data, val_data, \
            train_label, val_label = train_test_split(paths, labels, test_size=n_val)
        train_data, test_data, \
            train_label, test_label = train_test_split(train_data, train_label, test_size=n_test)
        return [train_data, train_label], \
            [val_data, val_label], \
            [test_data, test_label]

    train_pair, val_pair, test_pair = split_train_val_test(paths, labels, args.val, args.test)

    def transpose(pairs):
        return [[p, l] for p, l in zip(pairs[0], pairs[1])]
    
    def write_text(filename, pairs):
        mat = transpose(pairs)
        with open(filename, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerows(mat)


    write_text(train_txt,  train_pair)
    write_text(val_txt, val_pair)
    write_text(test_txt, test_pair)


    '''5. Create mean file'''
    dataset = train_classifier.MyLabeledImageDataset(transpose(train_pair))
    mean = compute_mean.compute_mean(dataset)
    np.save(os.path.join(args.out, 'mean'), mean)

    print('Complete!')
    
