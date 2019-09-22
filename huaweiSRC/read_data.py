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
        codes_path = os.getcwd() #代码存放地址
        
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
        
        
        father_path = os.path.abspath(os.path.join(os.getcwd(), "..")) #上一级目录
        train_set_path = father_path+'/train_set'
        try:
            if os.path.exists(train_set_path): #train数据集存放地址
                os.chdir( train_set_path )
                #print('In train_set directory!')
        except:
            print('Check train_set directory!')
        self.train_set_path = train_set_path
        os.chdir( codes_path ) #回到代码存放的地址

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
        clutter_index_dict = {1:1,2:1,3:1,4:1,5:1,6:1,7:2,8:2,9:2,10:6,11:5,
                              15:3,17:3,19:3,12:4,13:3,14:3,16:4,18:3,20:6 }
        def __direction(x,y):
            if x>=0 and y<0:
                return 180
            elif x<0 and y<0:
                return 180
            elif x<0 and y>=0:
                return 360
            else:
                return 0
        
        def __ConvertX(x):
            if x<0:
                return 0
            else:
                return x
        
        data.pop('Cell Index')
        data['New Clutter Index'] = data['Clutter Index'].map(lambda x: clutter_index_dict[x])
        data['New Cell Clutter Index'] = data['Cell Clutter Index'].map(lambda x: clutter_index_dict[x])
        data.pop('Clutter Index')
        data.pop('Cell Clutter Index')
    
        data['Distance'] = ((data['X'] - data['Cell X'])**2 + (data['Y'] - data['Cell Y'])**2)**0.5
        data['Delta_hv'] = data['Height']+data['Cell Altitude']-data['Building Height']-data['Altitude']-data['Distance']*((data['Electrical Downtilt'] + data['Mechanical Downtilt'])).map(lambda x: math.tan(math.radians(x)))
        #data['Cell_Height_Difference'] = (data['Cell Building Height'] - data['Height']).map(__ConvertX)
    
        data['RelativeX'] = data['X'] - data['Cell X']
        data['RelativeY'] = data['Y'] - data['Cell Y']
        #data['Direction_temp'] = (((data['X']-data['Cell X'])/(data['Y']-data['Cell Y'])).map(lambda x: math.atan(x)))*180/math.pi
        data['AddAngle'] = data.apply(lambda x: __direction(x.RelativeX,x.RelativeY),axis=1)
    
    
        data['Direction'] = ((((data['X']-data['Cell X'])/(data['Y']-data['Cell Y'])).map(lambda x: math.atan(x)))*180/math.pi + data['AddAngle'] - data['Azimuth']).map(lambda x: abs(x))
        data.pop('Cell X')
        data.pop('Cell Y')
        #data.pop('Height')
        data.pop('Azimuth')
        data.pop('Electrical Downtilt')
        data.pop('Mechanical Downtilt')
        data.pop('Cell Altitude')
        data.pop('Cell Building Height')
        data.pop('X')
        data.pop('Y')
        data.pop('Altitude')
        data.pop('Building Height')
        data.pop('Frequency Band')
    
        data.pop('RelativeX')
        data.pop('RelativeY')
        data.pop('AddAngle')
        #data.pop('Direction_temp')
        data.pop('Height')
        #data.pop('Cell_Height_Difference')
    
        if 'RSRP' in list(data):
            label = data.pop('RSRP')
            data['RS Power'] = data['RS Power'] / 10 ** np.ceil(np.log10(18.2))
            data['New Clutter Index'] = data['New Clutter Index']/10 ** np.ceil(np.log10(6))
            data['New Cell Clutter Index'] = data['New Clutter Index'] / 10 ** np.ceil(np.log10(6))
            data['Distance'] = data['Distance'] / 10 ** np.ceil(np.log10(5003.299))
            data['Delta_hv'] = data['Delta_hv'] /10**np.ceil(np.log10(471.05))
            data['Direction'] = data['Direction'] /10**np.ceil(np.log10(360))
            data.insert(6, 'RSRP', label)
        else:
            data['RS Power'] = data['RS Power'] / 10 ** np.ceil(np.log10(18.2))
            data['New Clutter Index'] = data['New Clutter Index'] / 10 ** np.ceil(np.log10(6))
            data['New Cell Clutter Index'] = data['New Clutter Index'] / 10 ** np.ceil(np.log10(6))
            data['Distance'] = data['Distance'] / 10 ** np.ceil(np.log10(5003.299))
            data['Delta_hv'] = data['Delta_hv'] / 10 ** np.ceil(np.log10(471.05))
            data['Direction'] = data['Direction'] / 10 ** np.ceil(np.log10(360))
        
    
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
                    if 'RSRP' in col_names:
                        inputs.append( a_file[[ _ for _ in col_names[0:-1]  ]] )
                        labels.append( a_file['RSRP'] )
                    else:
                        inputs.append( a_file[[ _ for _ in col_names  ]] )
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