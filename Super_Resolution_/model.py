
# coding: utf-8

# In[1]:

import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import os
from PIL import Image  # for loading images as YCbCr format
import matplotlib.pyplot as plt
#import h5py


# In[2]:

"""flags = tf.app.flags

flags.DEFINE_boolean("is_train", False, "[True] -> training, [False] -> testing")
FLAGS = flags.FLAGS"""


# In[3]:

#시험삼아 해본 cell, training에 전혀 무관
#data = misc.imread("./Train/tt1.bmp")
#data_resized = misc.imresize(data, (33, 33))#자동으로 (33x33)사이즈의 blur 이미지가 됨
def make_batch(input_data, label_data, batch_size):
    index = np.arange(0, len(input_data))
    np.random.shuffle(index)
    index = index[:batch_size]
    shuffled_input_data = [input_data[i] for i in index]
    shuffled_label_data = [label_data[i] for i in index]
    
    return np.asarray(shuffled_input_data), np.asarray(shuffled_label_data)


# In[4]:

class SRCNN(object):
    def __init__(self, sess, input_data, label_data, input_size, label_size, input_img_channel):
        self.sess = sess
        self.input_data = input_data
        self.label_data = label_data
        self.input_size = input_size#100
        self.label_size = label_size#120
        self.channel_dim = input_img_channel
        self.parameter()
        self.model()
        #self.training()
    
    
    def parameter(self):
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size, self.input_size, self.channel_dim])
        self.Y = tf.placeholder(dtype=tf.float32, shape=[None, self.label_size, self.label_size, self.channel_dim])
        #9x9크기, 3개의 채널, n1=64개의 필터를 가지는 커널.
        self.W1 = tf.Variable(tf.random_normal(shape=[9, 9, self.channel_dim, 64], stddev=1e-03))#shape=[9, 9, 3, 64]
        self.b1 = tf.Variable(tf.zeros(shape=[64]))
        
        self.W2 = tf.Variable(tf.random_normal(shape=[1, 1, 64, 32], stddev=1e-03))
        self.b2 = tf.Variable(tf.zeros(shape=[32]))
        
        self.W3 = tf.Variable(tf.random_normal(shape=[5, 5, 32, self.channel_dim], stddev=1e-03))
        self.b3 = tf.Variable(tf.zeros(shape=[self.channel_dim]))
        
    def model(self):
        #일단 1,1,1,1 로 stride 선택
        self.L1 = tf.nn.conv2d(self.X, self.W1, strides=[1, 1, 1, 1], padding="SAME")
        self.Y1 = tf.nn.relu(self.L1)
        
        self.L2 = tf.nn.conv2d(self.Y1, self.W2, strides=[1, 1, 1, 1], padding="SAME")
        self.Y2 = tf.nn.relu(self.L2)
        
        self.model = tf.nn.conv2d(self.Y2, self.W3, strides=[1, 1, 1, 1], padding="SAME")
        
        self.cost = tf.reduce_mean(tf.square(self.Y - self.model))
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.cost)
        
    def training(self, config):
        #3572 -> training 이미지 데이터 갯수, 32 -> batch size
        
        total_batch = int(3572 / 32)
        SAVE_PATH = "C:/Users/JAEKYU/Documents/Jupyter Notebook/Super_Resolution_/Weight/Weight.ckpt"
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #print(self.sess.run(self.W1))
        
        if(not config.is_train):
            saver.restore(self.sess, SAVE_PATH)
            print("weight 값 load 끝~")
            
        else:
            for epoch in range(15000):#15000
                total_cost = 0
                for i in range(total_batch):
                    #batch 선언
                    x_batch, y_batch = make_batch(self.input_data, self.label_data, batch_size=32)
                    _, cost_val = self.sess.run([self.optimizer, self.cost], feed_dict={self.X : x_batch, self.Y : y_batch})
                    total_cost = total_cost + cost_val

                cost_training = total_cost / total_batch
                #print("batch end")
                #plt.imshow(x_batch[0])
                #plt.show()
                #plt.imshow(y_batch[0])
                #plt.show()
                if((epoch % 100) == 0):
                    print("epoch : ", epoch+1, "cost : ", cost_training)
                    #label_data[1] = np.clip(label_data[1], 0, 255).astype(np.uint8)
                    #print(label_data[1])
                    self.testing([label_data[1]])
            saver.save(self.sess, SAVE_PATH)
            print("training 끝~")
        
    def testing(self, test_data):
        #print("testing...")
        predict = self.sess.run(self.model, feed_dict={self.X : test_data})
        predict = np.clip(predict, 0, 255).astype(np.uint8)
        #print(predict)
        #print(test_data)
        predict = np.array(predict)
        predict_buff = np.zeros((self.input_size, self.input_size, self.channel_dim))
        for row in range(self.input_size):
            for col in range(self.input_size):
                for channel in range(self.channel_dim):
                    predict_buff[row][col][channel] = -1 *  predict[0][row][col][channel]
                    
        
        #predict_buffer = predict.flatten()
        #predict = predict_buffer.reshape(120, 120, 3)
        #predict = misc.imresize(predict, (120, 120))#자동으로 (33x33)사이즈의 이미지가 됨
        plt.imshow(predict_buff)
        plt.show()


# In[5]:




# In[ ]:



