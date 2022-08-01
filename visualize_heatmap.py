import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import cv2
import numpy as np
import copy
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import *
from model import *
from readdata_VIGOR import DataLoader
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--area', type=str, help='same or cross area testing', default='same')
parser.add_argument('-i', '--img_idx', type=int, help='image index', default=110)
args = vars(parser.parse_args())
area = args['area']
img_idx = args['img_idx']

load_model_path = './model.ckpt' # path to the trained model
is_training = False
dimension = 8
train_test = 'test'

tf.reset_default_graph()
input_data = DataLoader(area, train_test)

dimension = 8
GT='Gaussian'

print('img_idx', img_idx)
tf.reset_default_graph()

# Define placeholers
sat = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat')
grd = tf.placeholder(tf.float32, [None, 320, 640, 3], name='grd')
gt = tf.placeholder(tf.float32, [None, 512, 512, 1], name='gt')

keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

batch_sat = np.zeros([1, input_data.sat_size[0], input_data.sat_size[1], 3], dtype=np.float32)
batch_grd = np.zeros([1, input_data.grd_size[0], input_data.grd_size[1], 3], dtype=np.float32)
batch_gt = np.zeros([1, input_data.sat_size[0], input_data.sat_size[1], 1], dtype=np.float32)

# ground
img = cv2.imread(input_data.val_list[img_idx])
img = img.astype(np.float32)
img = cv2.resize(img, (input_data.grd_size[1], input_data.grd_size[0]), interpolation=cv2.INTER_AREA)
img[:, :, 0] -= 103.939  # Blue
img[:, :, 1] -= 116.779  # Green
img[:, :, 2] -= 123.6  # Red
batch_grd[0, :, :, :] = img

pos_idx = 0 # use the positive satellite image. For semi-positives, change it to 1 or 2 or 3
img = cv2.imread(input_data.test_sat_list[input_data.val_label[img_idx][pos_idx]])
img = img.astype(np.float32)
img = cv2.resize(img, (input_data.sat_size[1], input_data.sat_size[0]), interpolation=cv2.INTER_AREA)
img[:, :, 0] -= 103.939  # Blue
img[:, :, 1] -= 116.779  # Green
img[:, :, 2] -= 123.6  # Red
batch_sat[0, :, :, :] = img

[col_offset, row_offset] = input_data.val_delta[img_idx, pos_idx] # delta = [delta_lat, delta_lon]
row_offset_resized = (row_offset/640*input_data.sat_size[0]).astype(np.int32)
col_offset_resized = (col_offset/640*input_data.sat_size[0]).astype(np.int32)
x, y = np.meshgrid(np.linspace(-input_data.sat_size[0]/2+row_offset_resized,input_data.sat_size[0]/2+row_offset_resized,input_data.sat_size[0]), np.linspace(-input_data.sat_size[0]/2-col_offset_resized,input_data.sat_size[0]/2-col_offset_resized,input_data.sat_size[0]))
d = np.sqrt(x*x+y*y)
sigma, mu = 4, 0.0
batch_gt[0, :, :, 0] = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )

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

    feed_dict = {sat: batch_sat, grd: batch_grd, keep_prob: 1.0, training: False}
    heatmap_val, matching_score_val = sess.run([heatmap, matching_score], feed_dict=feed_dict)  

batch_sat[0, :, :, 0] += 103.939  # Blue
batch_sat[0, :, :, 1] += 116.779  # Green
batch_sat[0, :, :, 2] += 123.6  # Red
batch_grd[0, :, :, 0] += 103.939  # Blue
batch_grd[0, :, :, 1] += 116.779  # Green
batch_grd[0, :, :, 2] += 123.6  # Red

plt.figure(figsize=(4,8))
plt.imshow(cv2.cvtColor(batch_grd[0, :, :, :], cv2.COLOR_BGR2RGB)/255)
plt.axis('off')
plt.savefig('results/figures/multi_ground_'+str(img_idx)+'.png',bbox_inches='tight')


plt.figure(figsize=(4,4))
plt.imshow(matching_score_val[0, :, :, 0])
plt.axis('off')
plt.savefig('results/figures/multi_matching_'+str(img_idx)+'.png',bbox_inches='tight')



plt.figure(figsize=(6,6))
plt.imshow(cv2.cvtColor(batch_sat[0, :, :, :], cv2.COLOR_BGR2RGB)/255)
plt.imshow(heatmap_val[0,:,:,0], norm=LogNorm(vmin=1e-9, vmax=np.max(heatmap_val[0,:,:,0])), alpha=0.6, cmap='Reds')
ax = plt.gca();
ax.set_xticks(np.arange(0, 512, 512/8));
ax.set_yticks(np.arange(0, 512, 512/8));
ax.grid(color='w', linestyle='-', linewidth=1)
loc_gt = np.unravel_index(batch_gt[0, :, :, :].argmax(), batch_gt[0, :, :, :].shape)
plt.scatter(loc_gt[1], loc_gt[0], s=200, marker='^', facecolor='g', label='GT', edgecolors='white')
loc_pred = np.unravel_index(heatmap_val[0, :, :, :].argmax(), heatmap_val[0, :, :, :].shape)
plt.scatter(loc_pred[1], loc_pred[0], s=200, marker='*', facecolor='gold', label='Ours', edgecolors='white')
plt.axis('off')
plt.savefig('results/figures/multi_heatmap_'+str(img_idx)+'.png',bbox_inches='tight')