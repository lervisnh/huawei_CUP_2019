# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:26:34 2019
@author: lervisnh
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import time
import csv
import tensorflow as tf
from tensorflow import layers
import numpy as np
from read_data import batch_data_generator

from tensorflow.python import pywrap_tensorflow
import warnings
warnings.filterwarnings('ignore')

class our_model:
    def __init__(self, input_dim, #输入维度
                 hidden_layers_info, #隐藏层，以字典格式保存信息
                                #层数：层类型、激活函数、神经元（filters）个数，其他（CNN）：kernel_size、strides
                 output_dim = 1, #输出维度
                 epoches = 200,  #训练最大总步数
                 minibatch_size = 10240, #mini batch训练大小
                 ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers_info = hidden_layers_info
        self.epoches = epoches
        self.minibatch_size = minibatch_size
        try: 
            self.codes_path = os.getcwd() #代码存放地址
            father_path = os.path.abspath(os.path.join(os.getcwd(), "..")) #上一级目录
            train_set_path = father_path+'/train_set'
            test_set_path = father_path+'/test_set'
            if os.path.exists(train_set_path) and os.path.exists(train_set_path): #train数据集存放地址
                self.train_set_path = train_set_path
                self.test_set_path = test_set_path
                print('Train_set and Test_set (validation_set) directory exists!')
        except:
            print('Train_set or Test_set (validation_set) directory doesn\'t exist, please CHECK!')
            
    '''
    def computing_features(self, inputs):
        clutter_index_dict = {1:0,2:0,3:0,4:0,5:0,6:0,7:1,8:1,9:1,10:5,
                              11:4,15:2,17:2,19:2,12:3,13:2,14:2,16:3,18:2,20:5 }
        cell_x, cell_y = inputs[:,1], inputs[:,2]
        height, azimuth = inputs[:,3], inputs[:,4]
        electrical_downtilt, mechanical_downtilt = inputs[:,5], inputs[:,6]
        frequency_band = inputs[:,7]
        rs_power = inputs[:,8]
        cell_altitude, cell_building_height, cell_clutter_index = inputs[:,9], inputs[:,10], inputs[:,11]
        x, y, altitude = inputs[:,12], inputs[:,13], inputs[:,14]
        building_height, clutter_index = inputs[:,15], inputs[:,16]
        
        cell_clutter_index = tf.map_fn( fn=lambda x:clutter_index_dict[x],
                                        elems=cell_clutter_index)
        clutter_index = tf.map_fn( fn=lambda x:clutter_index_dict[x],
                                   elems=clutter_index)
        
        distance = tf.sqrt( tf.add( tf.square( tf.subtract(cell_x, x) ), 
                                    tf.square( tf.subtract(cell_y, y) ) ) )
        
        relative_x, relative_y = tf.subtract(cell_x, x), tf.subtract(cell_y, y)
        
        delta_hv = height + cell_altitude - building_height - altitude - \
                   distance * tf.tan(electrical_downtilt + mechanical_downtilt)
        
        direction_temp = tf.multiply( tf.atan( (x-cell_x)/(y-cell_y) ), 180/math.pi)
        
        def direction__(x,y):
            if x>=0 and y<0:
                return 180
            elif x<0 and y<0:
                return 180
            elif x<0 and y>=0:
                return 360
            else:
                return 0

        add_angle = tf.map_fn( fn=direction__(), elems = (relative_x, relative_y) )
        
        direction = tf.abs( direction_temp + - azimuth )
    '''
    
    def building(self, input_layer):
        model_layers = [input_layer]
        for hidden_layer in self.hidden_layers_info: #关键字，层数1 2 3 4 5...
            this_layer_info = self.hidden_layers_info[hidden_layer]
            prev_layer = model_layers[-1]
            #普通层
            if this_layer_info['layer_type'] == 'Dense':
                prev_layer_shape = prev_layer.get_shape().as_list()
                if len(prev_layer_shape) == 3:
                    prev_layer = tf.reshape( prev_layer, (-1,prev_layer_shape[-1]) )
                    
                this_layer = layers.Dense( units = this_layer_info['units_filters'],
                                           activation = this_layer_info['activation'],
                                           name = 'hidden_'+str(hidden_layer)+'_'+\
                                                    this_layer_info['layer_type'])(prev_layer)
                model_layers.append( this_layer )
            #一维卷积层
            elif this_layer_info['layer_type'] == 'Conv1D':
                prev_layer_shape = prev_layer.get_shape().as_list()
                if len(prev_layer_shape) == 2:
                    prev_layer = tf.reshape( prev_layer, (-1,1,prev_layer_shape[-1]) )
                this_layer = layers.Conv1D( filters = this_layer_info['units_filters'],
                                            activation = this_layer_info['activation'],
                                            kernel_size = this_layer_info['kernel_size'],
                                            strides = this_layer_info['strides'],
                                            padding = 'same',
                                            name = 'hidden_'+str(hidden_layer)+'_'+\
                                                    this_layer_info['layer_type']
                                            )(prev_layer)
                model_layers.append( this_layer )
                
        last_hidden_layer = model_layers[-1]
        last_hidden_layer_shape = last_hidden_layer.get_shape().as_list()
        if len(last_hidden_layer_shape) > 2:
            last_hidden_layer_shape = tf.reshape( last_hidden_layer, 
                                                  (-1,last_hidden_layer_shape[-1]) )
        output_layer = layers.Dense(units = self.output_dim, name = 'output_layer')( last_hidden_layer )
        output = tf.multiply(output_layer, -100, name='magnification')
        return output
    
    
    def training(self):
        new_graph = tf.Graph()
        with new_graph.as_default() as new_g:
            #input占位符  模型输出
            input_layer = tf.placeholder(tf.float32, [None, self.input_dim], name = 'input_holder')
            output_layer = self.building(input_layer)
            #输出label占位符
            target_labels  = tf.placeholder(tf.float32, [None,self.output_dim], name = 'target_labels_holder')
            #目标函数
            rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(output_layer, target_labels)))
            # learning rate
            learning_rate = tf.train.exponential_decay(1.5e-4, self.epoches,
                                         decay_steps=self.epoches//10,
                                         decay_rate=0.9,staircase=True)
            solver = tf.train.AdamOptimizer(learning_rate).minimize(rmse)
            
        config = tf.ConfigProto(device_count = {'CPU': 4})
        with tf.Session(graph = new_g, config=config) as sess:
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init)
            
            # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
            epoches_val_rmse = [] #保存分批计算文件的验证RMSE
            epoches_mean_val_rmse = [] #保存分批计算文件的验证RMSE均值
            epoches_var_val_rmse = [] #保存分批计算文件的验证RMSE方差
            for epoch in range(self.epoches):
                epoch_time_start=time.time()
                # 迭代所有文件
                a_batch_data_generator = batch_data_generator(self.train_set_path, files_split_rate=.01)
                val_rmse = []
                for load_files, (inputs, labels) in enumerate(a_batch_data_generator.batch_generator()):
                    '''
                    #检查inputs
                    try:
                        if np.nan not in inputs:
                            pass
                        if np.inf not in inputs:
                            pass
                    except:
                        print( 'np.nan or np.inf in inputs !' )
                    #检查labels
                    try:
                        if np.nan not in labels:
                            pass
                        if np.inf not in labels:
                            pass
                    except:
                        print( 'np.nan or np.inf in labels !' )
                    '''
                    #乱序化输入输出对
                    _ = np.random.get_state()
                    np.random.shuffle(inputs)
                    np.random.set_state(_)
                    np.random.shuffle(labels)
                    #本次迭代读取文件验证集
                    val_samples_n = int(inputs.shape[0]*0.05)+1#取5%为验证集，辅助实现early stopping
                    val_inputs, val_labels = inputs[0:val_samples_n+1,:], labels[0:val_samples_n+1].reshape((-1,1))
                    #本次迭代读取文件训练集
                    inputs, labels = inputs[val_samples_n+1:,:], labels[val_samples_n+1:]
                    # mini batch
                    total_batch = int( inputs.shape[0] / self.minibatch_size ) #总批数
                    for i in range( total_batch+1 ): #range(total_batch+1) 或 range(total_batch)
                        batch_xs = inputs[ i*self.minibatch_size : (i+1)*self.minibatch_size , : ]
                        batch_ys = labels[ i*self.minibatch_size : (i+1)*self.minibatch_size].reshape((-1,1))
                        #sess.run( (output_layer), feed_dict={input_layer:batch_xs} )
                        _, c = sess.run((solver, rmse), 
                                        feed_dict={input_layer:batch_xs, target_labels:batch_ys})
                        #print('Mini Batch: ', i, ' / ', total_batch )
                    val_rmse.append( sess.run(rmse, feed_dict={input_layer:val_inputs, 
                                                               target_labels:val_labels}) )
                    #print('Load files.  No. ', load_files+1)
                    # 一次文件迭代完毕
                    
                epoches_val_rmse.append(val_rmse)
                epoches_mean_val_rmse.append(np.mean(val_rmse))
                epoches_var_val_rmse.append(np.var(val_rmse))
                # 所有训练集跑完一个epoch
                epoch_time_end=time.time()
                #一个epoch运行时间
                running_time_a_epoch = (epoch_time_end-epoch_time_start)/60
                print('================================================================')
                print("""Epoch:%d                Running time/epoch (min):%.3f""" 
                      %( epoch+1, running_time_a_epoch))
                print('--------     *      ------------------------     *      --------')
                print("""Validation RMSE mean:%.3f #  Validation RMSE var:%.3f""" 
                      %( epoches_mean_val_rmse[-1], epoches_var_val_rmse[-1]))
                #每 多少 个 epoch保存一下
                check_save_point = 3
                if not (epoch+1)%check_save_point:
                    #计算测试集
                    #test_batch_data_generator = batch_data_generator(self.test_set_path)
                    #test_inputs = test_batch_data_generator.merging()
                    #test_predictions = sess.run(output_layer, feed_dict={input_layer:test_inputs})
                    tf.saved_model.simple_save(sess, self.codes_path+'/model-epoch-'+str(epoch), 
                                           inputs={"myInput": input_layer}, 
                                           outputs={"myOutput": output_layer})
                    
                #最近训练的 patience 次，误差在 min_delta 内
                min_delta = 0.1
                patience = 5
                if epoch > patience and \
                   abs( np.mean(epoches_mean_val_rmse[-1-patience:-1])
                        -epoches_mean_val_rmse[-1] ) < min_delta:
                    print('################################################################')
                    print('!!!  Early stopping  !!!')
                    print('Min delta:', min_delta)
                    print('Patience:', patience)
                    #保存最优模型模型
                    tf.saved_model.simple_save(sess, self.codes_path+'/model'+'-best', 
                                           inputs={"myInput": input_layer}, 
                                           outputs={"myOutput": output_layer})
                    print('--------     *      ------------------------     *      --------')
                    print('Path to save model: ', self.codes_path+'/model')
                    print('--------     *      ------------------------     *      --------')
                    
                    #计算测试集
                    test_batch_data_generator = batch_data_generator(self.test_set_path)
                    test_inputs = test_batch_data_generator.merging()
                    test_predictions = sess.run(output_layer, feed_dict={input_layer:test_inputs})
                    
                    break #早停            
                    
            #写出epoches_val_rmse
            with open(self.codes_path+'/val_rmse_records.csv', 'w', newline='') as csvfile:
                writer  = csv.writer(csvfile)
                for row in epoches_val_rmse:
                    writer.writerow(row)
            #保存模型测试结果
            np.savetxt(self.codes_path+'/test_predictions.csv', test_predictions, delimiter=',')  
            #递交sess
            self.session =  sess

    #保存为ckpt
    def saving(self, sess, path_to_save):
        saver = tf.train.Saver()
        saver.save(sess, path_to_save)
    
    #预测
    def predicting(self):
        #合并test的数据集
        a_batch_data_generator = batch_data_generator(self.test_set_path)
        test_inputs = a_batch_data_generator.merging()
        #input占位符  模型输出
        inputs = tf.placeholder(tf.float32, [None, self.input_dim], name = 'input_predicting_hoder')
        predictions = self.building(inputs)
        if 'session' in dir(): #已经包含了session，即训练过了
            with self.session as sess:
                prediction = sess.run(predictions, feed_dict={inputs:test_inputs})
                tf.saved_model.simple_save(sess, self.codes_path+'/model', 
                                           inputs={"myInput": inputs}, 
                                           outputs={"myOutput": predictions})
        else:
            with tf.Session() as sess:
                model_f = tf.gfile.FastGFile("/model/saved_model.pb", mode='rb')
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(model_f.read())
                input_layer = tf.import_graph_def( graph_def, return_elements=["input_holder:0"])
                output_layer = tf.import_graph_def( graph_def, return_elements=["magnification:0"] )
                prediction = sess.run(output_layer, feed_dict={input_layer:test_inputs})
                '''
                # tf.initialize_all_variables() no long valid from
                # 2017-03-02 if using tensorflow >= 0.12
                if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0])<1:
                    init = tf.initialize_all_variables()
                else:
                    init = tf.global_variables_initializer()
                sess.run(init)
                path_to_save = self.codes_path+'/model'
                restoration = tf.train.import_meta_graph(path_to_save+'/variables/variables.meta')
                restoration.restore(sess, tf.train.latest_checkpoint(path_to_save))
                restoration.restore(sess, os.path.join(path_to_save+'/variables', 
                                                      'variables'))
                prediction = sess.run(predictions, feed_dict={inputs:test_inputs})
                '''
        return prediction
    
    
    def ckpt_to_pd(self, pd_name):
        with tf.Session() as sess:
            # tf.initialize_all_variables() no long valid from
            # 2017-03-02 if using tensorflow >= 0.12
            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0])<1:
                init = tf.initialize_all_variables()
            else:
                init = tf.global_variables_initializer()
            sess.run(init) #初始化session
            path_to_save = self.codes_path+'/saving_model'
            restoration = tf.train.import_meta_graph(path_to_save+'/variables.meta')#载入图模型
            restoration.restore(sess, tf.train.latest_checkpoint(path_to_save))#载入图模型的参数
            
            input_graph_def = tf.get_default_graph().as_graph_def()
            #node_names = [n.name for n  in input_graph_def.node]
            
            checkpoint_path = self.codes_path+'/saving_model/variables'
            reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
            output_node_names = list(reader.get_variable_to_shape_map().keys())
                
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                             input_graph_def=input_graph_def,
                                                             output_node_names=output_node_names )
            output_graph = 'saving_model/'+pd_name+'.pd'
            with tf.gfile.GFile(output_graph, 'wb') as fw:
    	           fw.write(output_graph_def.SerializeToString())
            print ('{} ops in the final graph.'.format(len(output_graph_def.node)))

                
    
    
if  __name__=='__main__':
    hidden_layers_info = {1:{'layer_type':'Dense', 
                             'activation':'tanh',
                             'units_filters':10,
                             'kernel_size':4,
                             'strides':2},     
                          2:{'layer_type':'Dense', 
                             'activation':'relu',
                             'units_filters':15,
                             'kernel_size':6,
                             'strides':2},
                          3:{'layer_type':'Dense', 
                             'activation':'relu',
                             'units_filters':8,
                             'kernel_size':4,
                             'strides':2},
                          4:{'layer_type':'Dense', 
                             'activation':'sigmoid',
                             'units_filters':6,
                             'kernel_size':4,
                             'strides':2},}
    
    our_one_model = our_model(input_dim = 6, 
                              hidden_layers_info=hidden_layers_info, 
                              epoches = 200,
                              minibatch_size = 2048 )
    
    our_one_model.training()
    predictions = our_one_model.predicting()
    #our_one_model.ckpt_to_pd(pd_name = 'export_pd_9.22.morning')