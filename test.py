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
import pandas as pd
from train_classifier import *
from chainer import Variable
import cupy

h = Variable(np.array([[0.2, 0.1, 1.0], [0.9, 0.0, 0.8]], dtype=np.float32))
t = cupy.array([[1, 0, 1], [1, 0, 0]], dtype=np.int32)

accuracy = evaluate_accuracy(h, t, suggestion=2)
#accuracy = F.classification_summary(h, t)
#print(t/h)
print(accuracy)
