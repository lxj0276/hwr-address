# coding: utf-8
# keyword_recognizer.py

import cnn
import sys
sys.path.append('../data_set/hcl/')
from hcl import input_data

hcl = input_data(['省','市','县','区','乡','镇','村','巷','弄','路', '街', '社', '组', '队', '州', 'X'], 500, 200, True, True, (32, 32), False)

x_shape = 1024
cnn_reshape = [-1,32,32,1]
y_shape = 16
cnn_layer_n = 2
cnn_weights = [[3, 3, 1, 32], [3, 3, 32, 64]]
keep_prob = [1, 1, 1, 1, 0.5]
fnn_reshape = [-1, 8*8*64]
fnn_layer_n = 1
fnn_weights = [[8*8*64, 1024]]
softmax_weight = [1024, 16]
a = cnn(x_shape, cnn_reshape, y_shape, cnn_layer_n, cnn_weights, keep_prob, fnn_reshape, fnn_layer_n, fnn_weights, softmax_weight)
a.train(hcl, 2000, 50)
a.test(hcl, 200)
