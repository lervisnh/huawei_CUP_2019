# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:07:22 2019
@author: lervisnh
"""
import os
import numpy as np
import pandas as pd
import random
import math

class batch_data_generator:
    def __init__(self, data_files_path, files_split_rate=.05):
        try:
            if os.path.exists(data_files_path): #train数据集存放地址
                os.chdir( data_files_path )
                #print('In '+data_files_path.split('\\')[-1]+' directory!')
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
        
        
        codes_path = os.getcwd() #代码存放地址
        father_path = os.path.abspath(os.path.join(os.getcwd(), "..")) #上一级目录
        train_set_path = father_path+'/train_set'
        try:
            if os.path.exists(train_set_path): #train数据集存放地址
                os.chdir( train_set_path )
                #print('In train_set directory!')
        except:
            print('Check train_set directory!')
        self.train_set_path = train_set_path
        

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
                        self.create_features(a_file)
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
                        self.create_features(a_file)
                        #这里可以加特征预处理
                        ####################
                        col_names = list(a_file)
                        inputs.append( a_file[[ _ for _ in col_names[0:-1]  ]] )
                        labels.append( a_file['RSRP'] )
                yield (np.concatenate(inputs), np.concatenate(labels))

    def create_features(self, data):
        cluter_index_map = {1:0, 2:0, 3:0, \
                                4:0, 5:0, 6:0, 7:1, 8:1, \
                                9:1, 10:5, 11:4, 15:2, 17:2, 19:2, \
                                12:3, 13:2, 14:2, 16:3, 18:2, \
                                20:5 }
        '''
        p = data.describe()
        # 过滤异常数据
        up_limit = p.loc['75%', 'RSRP'] + 1.5 * (p.loc['75%', 'RSRP'] - p.loc['25%', 'RSRP'])
        down_limit = p.loc['25%', 'RSRP'] - 1.5 * (p.loc['75%', 'RSRP'] - p.loc['25%', 'RSRP'])
        data[u'RSRP'][(data[u'RSRP'] < down_limit)] = np.nan
        data[u'RSRP'][data[u'RSRP'] > up_limit] = np.nan
        data.dropna(axis=0, how='any')
        '''
        # 删除不需要的列
        del data['Cell Index']
    
        data['Distance'] = ((data['X'] - data['Cell X']) ** 2 + (data['Y'] - data['Cell Y']) ** 2) ** 0.5
        data['Clutter Index'].map(cluter_index_map)
        data['Cell Clutter Index'].map(cluter_index_map)
        data['Delta_hv'] = data['Height'] + data['Cell Altitude'] - data['Building Height'] - data['Altitude'] - data[
            'Distance'] * ((data['Electrical Downtilt'] + data['Mechanical Downtilt'])).map(
            lambda x: math.tan(math.radians(x)))
        data['Cell_Height_Difference'] = data['Cell Building Height'] - data['Height']
        data['Direction'] = (((data['X'] - data['Cell X']) / (data['Y'] - data['Cell Y'])).map(
            lambda x: math.atan(x))) * 180 / math.pi / data['Azimuth']
        del data['Cell X']
        del data['Cell Y']
        del data['Height']
        del data['Azimuth']
        del data['Electrical Downtilt']
        del data['Mechanical Downtilt']
        del data['Cell Altitude']
        del data['Cell Building Height']
        del data['X']
        del data['Y']
        del data['Altitude']
        del data['Building Height']
        if 'RSRP' in list(data):
            label = data.pop('RSRP')
            data.insert(8, 'RSRP', label)
        del data['Frequency Band']
        del data['Cell_Height_Difference']
    
        #return data
    
    
    def merging(self):
        files_name = os.listdir(self.data_files_path) #得到文件夹下的所有文件名称
        inputs, labels = [], []
        for file_name in files_name:
            with open(self.data_files_path +'/'+ file_name, 'r') as data_file:
                    a_file = pd.read_csv( data_file )
                    ####################
                    self.create_features(a_file)
                    #这里可以加特征预处理
                    ####################
                    col_names = list(a_file)
                    inputs.append( a_file[[ _ for _ in col_names[0:-1]  ]] )
                    if 'RSRP' in col_names:
                        labels.append( a_file['RSRP'] )
        if 'RSRP' in col_names:
            return (np.concatenate(inputs), np.concatenate(labels))
        else:
            return np.concatenate(inputs)

if __name__=='__main__':
    codes_path = os.getcwd() #代码存放地址
    father_path = os.path.abspath(os.path.join(os.getcwd(), "..")) #上一级目录
    train_set_path = father_path+'/train_set'
    test_set_path = father_path+'/test_set'
    try:
        if os.path.exists(train_set_path) and os.path.exists(train_set_path): #train数据集存放地址
            print('Train_set and Test_set (validation_set) directory exists!')
    except:
            print('Train_set or Test_set (validation_set) directory doesn\'t exist, please CHECK!')
            
    #合并test的数据集
    a_batch_data_generator = batch_data_generator(test_set_path)
    test_inputs = a_batch_data_generator.merging()

    #迭代输出train data
    batch_data_generator = batch_data_generator(train_set_path, files_split_rate=.01)

    n = 1
    for x, y in batch_data_generator.batch_generator() :
        print(n)
        print(x.shape)
        print(y.shape)
        n+=1