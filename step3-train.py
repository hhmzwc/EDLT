# -*- coding: utf-8 -*-
"""


@author: Han
"""

from sklearn import datasets
import tensorflow as tf
import numpy as np
import math
import os
from collections import Counter
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
# from sklearn.datasets import fetch_
from sklearn.feature_selection import chi2
import tensorflow.contrib.slim as slim
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import tempfile
from six.moves import urllib
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
seed =7
filter_size=2
whol_dataset=1
###the number of full connected layer
onehidden=0
tf.set_random_seed(1)
def get_random_block_from_data(data_xs, data_ys, batch_size,whol_dataset):
  if (whol_dataset==0):
    start_index = np.random.randint(0, len(data_xs) - batch_size)
    xs_data=data_xs[start_index:(start_index + batch_size)]
    ys_data=data_ys[start_index:(start_index + batch_size)]
  else:
    xs_data=data_xs
    ys_data=data_ys
  return xs_data,ys_data
def make_one_hot(data1):
    return (np.arange(label_size)==data1[:,None]).astype(np.integer)

num_epochs = 180
CNN_layer=[32,16]
layer_number=2
average_split_cnn=[]



path = './dataset'
dirs = os.listdir(path)
random_state_total = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
dataset_num=0
for dir in dirs:

    ini_data = (np.load('./dataset/' + dir + '/ini_data.npy'))
    ini_target = (np.load('./dataset/' + dir + '/ini_target.npy'))
    n_feature = ini_data.shape[1]
    ini_target_dif = set(ini_target)
    label_size = len(ini_target_dif)
    if onehidden == 1:
        full_layer = [100, label_size]
        full_layer_number = 2
    else:
        full_layer = [100, 50, label_size]
        full_layer_number = 3
    feature_whole_map_one_line=(np.loadtxt('./dataset/' + dir + '/feature_map/filter4/feature_map_one_line.dat',delimiter=','))
    feature_map_index=0
    for tt in range(len(random_state_total)):
        print('tt',tt)
        skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=random_state_total[tt])
        for train_index, test_index in skf.split(ini_data, ini_target):
            train_data1, test_data1 = ini_data[train_index], ini_data[test_index]
            train_label1, test_label1 = ini_target[train_index], ini_target[test_index]
            min_max_scaler = preprocessing.MinMaxScaler()
            scaler_data_train = min_max_scaler.fit_transform(train_data1)
            scaler_data_test = min_max_scaler.transform(test_data1)

            cor_score=[]
            cor=[]
            if (whol_dataset==1):
              batch_size = len(train_data1)
            else:
              batch_size = 128

            #feature reordering matrix: feature_map
            feature_map=np.zeros([n_feature,n_feature])
            for ii in range(n_feature):
                for jj in range(n_feature):
                    feature_map[ii][jj]=feature_whole_map_one_line[feature_map_index]
                    feature_map_index=feature_map_index+1

            # get train data all
            new_instance_train=np.zeros([len(train_data1),n_feature,n_feature])
            for kk in range(len(train_data1)):
                for ll in range(n_feature):
                    for mm in range(n_feature):
                        index=np.array(feature_map[ll,mm].copy()-1,dtype=int)
                        feature_new=scaler_data_train[kk,index].copy()
                        new_instance_train[kk,ll,mm]=feature_new
            new_instance_train=np.reshape(new_instance_train,[len(train_data1),n_feature,n_feature,1])
            train_data=new_instance_train.copy()
            train_label=make_one_hot(train_label1)
            #
            #get test data
            new_instance_test=np.zeros([len(test_data1),n_feature,n_feature])
            for kk in range(len(test_data1)):
              # for bb in range(n_feature):
                for ll in range(n_feature):
                    for mm in range(n_feature):
                        index=np.array(feature_map[ll,mm].copy()-1,dtype=int)
                        feature_new=scaler_data_test[kk,index].copy()
                        new_instance_test[kk,ll,mm]=feature_new
            new_instance_test=np.reshape(new_instance_test,[len(test_data1),n_feature,n_feature,1])
            test_data=new_instance_test.copy()
            test_label=make_one_hot(test_label1)



            ########CNN classification (feature_extraction and then selection)
            tf.reset_default_graph()
            graph=tf.Graph()
            with graph.as_default() as g:
                inputs = tf.placeholder(tf.float32, shape=(None, n_feature,n_feature,1))
                outputs = tf.placeholder(tf.float32)
                def CNN_feature_selection_and_full(inputs):
                    current_input=inputs
                    for i in range(layer_number):
                        cov_1 = slim.conv2d(current_input, CNN_layer[i], [filter_size,filter_size])
                        active_cov_1=slim.nn.leaky_relu(cov_1)
                        max_pool_1= slim.max_pool2d(active_cov_1, kernel_size=[2, 2], stride=2)
                        current_input=max_pool_1
                    out_feature=slim.flatten(current_input)
                    full_input=out_feature
                    for kk in range (full_layer_number):
                     full_connect=slim.fully_connected(full_input, full_layer[kk], activation_fn=None,normalizer_fn=None)
                     if(kk==full_layer_number-1):
                       full_input=full_connect
                     else:
                         full_input = slim.nn.leaky_relu(full_connect)
                    out = (full_input)
                    return out
                classifier_out=CNN_feature_selection_and_full(inputs)


                # LOSS AND OPTIMIZER
                cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=outputs, logits=classifier_out))
                optm = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
                corr = tf.equal(tf.argmax(tf.nn.softmax(classifier_out), 1), tf.argmax(outputs, 1))
                accr = tf.reduce_mean(tf.cast(corr, "float"))* tf.constant(100.0)
                saver = tf.train.Saver()

                print( "===  Starting Session ===")
                sess = tf.Session()
                i_iter = 0
                init = tf.global_variables_initializer()
                sess.run(init)
                print ("=== Training ===")
                print ("Initial Accuracy: ", sess.run(accr, feed_dict={inputs: test_data, outputs: test_label}), "%")
                for i in tqdm(range(i_iter, num_epochs)):
                 per_train_iter=int(len(train_data)/batch_size)
                 for mm in range(per_train_iter):
                    batch_xs, batch_ys= get_random_block_from_data(train_data, train_label,batch_size,whol_dataset)
                    sess.run(optm, feed_dict={inputs: batch_xs, outputs: batch_ys})
                 print ("Final Accuracy: ", sess.run(accr, feed_dict={inputs: test_data, outputs: test_label}), "%")
                accu=sess.run(accr, feed_dict={inputs: test_data, outputs: test_label})
                average_split_cnn.append(accu)
                sess.close()
    file_handle = open('./dataset/' + dir + '/test_result/test_cnn_our_32_16_f2_l2.txt', mode='a')
    file_handle.write('configure\n')
    file_handle.write(
        '%s%d%s%d%s%d%s%d%d%s%d\n\n' % ("num_epochs:", num_epochs, "hidden_size:", onehidden, "layer_num:", layer_number,"CNN_layer:", CNN_layer[0],CNN_layer[1],"filter_size:",filter_size))
    file_handle.write('result\n')
    file_handle.write('%s%f\n\n' % ("cnn-order:",np.mean(np.array(average_split_cnn))))
    print('average_split_cnn',average_split_cnn)
    file_handle.close()





