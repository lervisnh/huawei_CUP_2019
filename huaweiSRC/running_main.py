# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:19:26 2019

@author: lervisnh
"""
from dl_model import our_model

if  __name__=='__main__':
    hidden_layers_info = {1:{'layer_type':'Dense', 'activation':'relu', 'units_filters':15 },
                          2:{'layer_type':'Dense', 'activation':'relu', 'units_filters':15, \
                                                                                 'kernel_size':4, \
                                                                                 'strides':2},
                          3:{'layer_type':'Dense', 'activation':'tanh', 'units_filters':15 },
                          4:{'layer_type':'Dense', 'activation':'tanh', 'units_filters':10 },
                          5:{'layer_type':'Dense', 'activation':'tanh', 'units_filters':10 },
                          6:{'layer_type':'Dense', 'activation':'tanh', 'units_filters':10 },}
    our_one_model = our_model(6, hidden_layers_info )
    
    our_one_model.training()