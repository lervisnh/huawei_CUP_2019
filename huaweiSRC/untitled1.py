# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:26:34 2019
@author: lervisnh
"""

from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Model, save_model, load_model, Sequential
import keras.layers as layers
from keras import optimizers
from keras.layers import Input, Dense, Conv1D, Flatten, Dropout, concatenate, Reshape
from keras.utils import plot_model
import numpy as np
K.tensorflow_backend._get_available_gpus()#使用GPU

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util

#输入训练数据 keras接收numpy数组类型的数据
x=np.array([[0,1,0],
            [0,0,1],
            [1,3,2],
            [3,2,1]])
y=np.array([0,0,1,1]).T
#最简单的序贯模型，序贯模型是多个网络层的线性堆叠
simple_model=Sequential()
#dense层为全连接层
#第一层隐含层为全连接层 5个神经元 输入数据的维度为3
simple_model.add(Dense(5,input_dim=3,activation='relu', name='NN1'))
#第二个隐含层 4个神经元
simple_model.add(Dense(4,activation='relu', name='NN2'))
#输出层为1个神经元
simple_model.add(Dense(1,activation='sigmoid', name='output'))
#编译模型,训练模型之前需要编译模型
#编译模型的三个参数：优化器、损失函数、指标列表
simple_model.compile(optimizer='sgd',loss='mean_squared_error')
#训练网络 2000次
#Keras以Numpy数组作为输入数据和标签的数据类型。训练模型一般使用fit函数
simple_model.fit(x,y,epochs=2)

sess = K.get_session()
save_model( simple_model,  'saving_models/keras_models.h5' )
saver = tf.train.Saver()
save_path = saver.save(sess, 'saving_models/tf_models')

loaded_model = load_model( 'saving_models/keras_models.h5'  )
loaded_model.summary()

'''
import os

def read_graph_from_ckpt(ckpt_path, out_pb_path, output_name ):     
    # 从meta文件加载网络结构
    saver = tf.train.import_meta_graph(ckpt_path+'.meta',clear_devices=True)
    graph = tf.get_default_graph()
    with tf.Session( graph=graph) as sess:
        sess.run(tf.global_variables_initializer()) 
        # 从ckpt加载参数
        saver.restore(sess, ckpt_path) 
        output_tf =graph.get_tensor_by_name(output_name) 
        
        # 固化
        pb_graph = tf.graph_util.convert_variables_to_constants( sess, graph.as_graph_def(), [output_tf.op.name]) 
    
        # 保存
        with tf.gfile.FastGFile(out_pb_path, mode='wb') as f:
          f.write(pb_graph.SerializeToString())
     
read_graph_from_ckpt('saving_models/tf_models.ckpt', \
                     'tf_models.pb',  'op_to_store:1')
'''