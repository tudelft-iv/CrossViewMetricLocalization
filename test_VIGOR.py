import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# os.environ["MKL_NUM_THREADS"] = "3" 
# os.environ["NUMEXPR_NUM_THREADS"] = "3" 
# os.environ["OMP_NUM_THREADS"] = "3" 
import cv2
import random
import numpy as np
import copy
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import *
from model import *
from readdata_VIGOR import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--area', type=str, help='same or cross area testing', required=True)
args = vars(parser.parse_args())
area = args['area']

load_model_path = './model.ckpt' # path to the trained model
batch_size = 32
is_training = False
dimension = 8
train_test = 'test'

tf.reset_default_graph()
input_data = DataLoader(area, train_test)

# Define placeholers
sat = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat')
grd = tf.placeholder(tf.float32, [None, 320, 640, 3], name='grd')
gt = tf.placeholder(tf.float32, [None, 512, 512, 1], name='gt')

keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

# Build model
logits, matching_score = CVML(sat, grd, keep_prob, dimension, is_training, False)
logits_reshaped = tf.reshape(logits, [tf.shape(logits)[0], 512*512])
heatmap = tf.reshape(tf.nn.softmax(logits_reshaped), tf.shape(logits))

saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('load model...')
    saver.restore(sess, load_model_path)
    
    input_data.reset_scan()
    distance = []
    probability = []
    probability_at_gt = []
    processed_batch = 0 
    while True:
        batch_sat, batch_grd, batch_gt = input_data.next_batch_scan(batch_size)
        if batch_sat is None:
            break
        processed_batch += batch_sat.shape[0]
        print('processed_batch: ', processed_batch)
        feed_dict = {sat: batch_sat, grd: batch_grd, keep_prob: 1.0, training: False}
        heatmap_val = sess.run(heatmap, feed_dict=feed_dict)
        for batch_idx in range(batch_gt.shape[0]):
            current_gt = batch_gt[batch_idx, :, :, :]
            loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
            current_pred = heatmap_val[batch_idx, :, :, :]
            
            loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
            distance.append(np.sqrt((loc_gt[0]-loc_pred[0])**2+(loc_gt[1]-loc_pred[1])**2)) 
            probability_at_gt.append(current_pred[loc_gt[0], loc_gt[1]])

print('mean distance error', np.mean(distance)*0.1425) # 0.1425 is the ground distance per pixel. 0.1425 = 0.114/512*640
print('median distance error', np.median(distance)*0.1425)
print('mean distance error', np.mean(probability_at_gt))
print('mean distance error', np.median(probability_at_gt))