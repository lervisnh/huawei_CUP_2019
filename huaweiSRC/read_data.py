# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:07:22 2019
@author: lervisnh
"""
import os
import numpy as np
import pandas as pd
import random

class batch_data_generator:
    def __init__(self, data_files_path, files_split_rate=.05):
        try:
            if os.path.exists(data_files_path): #train数据集存放地址
                os.chdir( data_files_path )
                print('In '+data_files_path.split('\\')[-1]+' directory!')
        except:
            print('Check '+data_files_path.split('\\')[-1]+' directory!')
        #  data_files_path为存放数据的绝对路径
        self.data_files_path = data_files_path
        #所有文件的名字
        self.files_name = os.listdir(data_files_path) #得到文件夹下的所有文件名称
        #每次读取多少比例的文件
        self.files_split_rate = files_split_rate
        #每次迭代文件的数量（最大）
        self.files_n_per_generator = int(len(self.files_name)*self.files_split_rate)+1

    def batch_generator(self):
        while self.files_name:#仍有未读取的文件
            inputs, labels = [], []
            #剩余文件数量大于每次迭代文件数量
            if len(self.files_name) > self.files_n_per_generator: 
                for _ in range(self.files_n_per_generator):
                    index = random.randint(0,len(self.files_name)-1)
                    file_name = self.files_name.pop(index)
                    with open(self.data_files_path +'/'+ file_name, 'r') as data_file:
                        a_file = pd.read_csv( data_file )
                        ####################
                        #这里可以加特征预处理
                        ####################
                        col_names = list(a_file)
                        inputs.append( a_file[[ _ for _ in col_names[0:-1]  ]] )
                        labels.append( a_file['RSRP'] )
                yield (np.concatenate(inputs), np.concatenate(labels))
            #（剩余）文件数量不足每次迭代文件数量，则全部取出
            else: 
                while self.files_name: #读取所有剩余的文件
                    file_name = self.files_name.pop(-1)
                    with open(self.data_files_path +'/'+ file_name, 'r') as data_file:
                        a_file = pd.read_csv( data_file )
                        ####################
                        #这里可以加特征预处理
                        ####################
                        col_names = list(a_file)
                        inputs.append( a_file[[ _ for _ in col_names[0:-1]  ]] )
                        labels.append( a_file['RSRP'] )
                yield (np.concatenate(inputs), np.concatenate(labels))
        
    def create_features(self, ):
        pass
    


if __name__=='__main__':
    codes_path = os.getcwd() #代码存放地址
    father_path = os.path.abspath(os.path.join(os.getcwd(), "..")) #上一级目录
    train_set_path = father_path+'/train_set'
    try:
        if os.path.exists(train_set_path): #train数据集存放地址
            os.chdir( train_set_path )
            print('In train_set directory!')
    except:
        print('Check train_set directory!')
    batch_data_generator = batch_data_generator(train_set_path, files_split_rate=.05)
    
    n = 1
    for x, y in batch_data_generator.batch_generator() :
        print(n)
        print(x.shape)
        print(y.shape)
        n+=1