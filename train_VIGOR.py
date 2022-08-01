import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
os.environ["MKL_NUM_THREADS"] = "3" 
os.environ["NUMEXPR_NUM_THREADS"] = "3" 
os.environ["OMP_NUM_THREADS"] = "3" 
import cv2
import random
import numpy as np
import copy
import scipy
import scipy.stats as stats
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import *
from model import CVML
from readdata_VIGOR import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-a', '--area', type=str, help='same or cross area testing', required=True)
parser.add_argument('-l', '--learning_rate', type=float, help='learning rate', default=1e-5)
parser.add_argument('-e', '--start_epoch', type=int, help='start epoch', default=0)
parser.add_argument('-b', '--batch_size', type=int, help='batch size', default=8)
args = vars(parser.parse_args())
area = args['area']
learning_rate_val = args['learning_rate']
start_epoch = args['start_epoch']
batch_size = args['batch_size']

is_training = True
number_of_epoch = 30
keep_prob_val = 0.8
dimension = 8
beta = 1e4
temperature=0.1
label = 'VIGOR_'+area
save_model_path = './models/'

def contrastive_loss(scores, labels, temperature=1.0):
    """Contrastive loss over matching score. Adapted from https://arxiv.org/pdf/2004.11362.pdf Eq.2
    We extraly weigh the positive samples using the ground truth likelihood on those positions
    
    loss = - 1/sum(weights) * sum(inner_element*weights)
    inner_element = log( exp(score_pos/temperature) / sum(exp(score/temperature)) )
    """
        
    exp_scores = tf.math.exp(scores / temperature)
    bool_mask = tf.cast(labels>1e-2, tf.bool) # only keep positive samples, we set a threshod on the likelihood in GT
    denominator = tf.reduce_sum(exp_scores, [1, 2, 3], keepdims=True)
    
    inner_element = tf.math.log(tf.boolean_mask(exp_scores/denominator, bool_mask)) 

    return -tf.reduce_sum(inner_element*tf.boolean_mask(labels, bool_mask)) / tf.reduce_sum(tf.boolean_mask(labels, bool_mask))
    

def train(start_epoch=0, learning_rate_val=learning_rate_val):
    '''
    Train the model from epoch N. Default is 0.
    '''
    # Import data
    input_data = DataLoader(area, 'train')
    
    # Define placeholers
    sat = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat')
    grd = tf.placeholder(tf.float32, [None, 320, 640, 3], name='grd')
    gt = tf.placeholder(tf.float32, [None, 512, 512, 1], name='gt')
    gt_bottleneck = tf.nn.max_pool(gt, ksize=[1, 64, 64, 1], strides=[1, 64, 64, 1], padding='SAME', name='gt_bottleneck')
   
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    tf.summary.scalar('learning_rate', learning_rate)
    training = tf.placeholder(tf.bool)
    
    # Build model
    logits, matching_score = CVML(sat, grd, keep_prob, dimension, is_training, False)
    logits_reshaped = tf.reshape(logits, [tf.shape(logits)[0], 512*512])
    
    gt_reshaped = tf.reshape(gt, [tf.shape(logits)[0], 512*512])
    gt_reshaped = gt_reshaped / tf.reduce_sum(gt_reshaped, axis=1, keepdims=True)
    loss_heatmap = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt_reshaped, logits=logits_reshaped))
    loss_bottleneck = contrastive_loss(matching_score, gt_bottleneck, temperature)
    loss_bottleneck_summary = tf.summary.scalar('loss_bottleneck', loss_bottleneck)
    
    loss = loss_heatmap + loss_bottleneck*beta
    loss_summary = tf.summary.scalar('loss', loss)
    
    heatmap = tf.reshape(tf.nn.softmax(logits_reshaped), tf.shape(logits))

    # Get all summaries
    summary = tf.summary.merge_all()
    
    # set training
    global_step = tf.Variable(0, trainable=False)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    
    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        print('load model...')

        if start_epoch == 0:
            load_model_path_init = save_model_path+'Initialize/initial_model.ckpt'
            variables_to_restore_init = tf.contrib.framework.get_variables_to_restore(include=['VGG_grd','VGG_sat'])
            init_fn = tf.contrib.framework.assign_from_checkpoint_fn(load_model_path_init, variables_to_restore_init)
            init_fn(sess)
            print("   Model initialized from: %s" % load_model_path_init)
        else:
            load_model_path = save_model_path+label+'/' + str(start_epoch - 1) + '/model.ckpt'
            saver.restore(sess, load_model_path)
            print("   Model loaded from: %s" % load_model_path)
            print('load model...FINISHED')    
        
        # Define tensorboard writer
        logdir = './graph/'+label
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        graph = tf.get_default_graph()
        writer = tf.summary.FileWriter(logdir, sess.graph)
        
        # Train
        for epoch in range(start_epoch, number_of_epoch):
            iter = 0
            while True:
                batch_sat, batch_grd, batch_gt = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break
                    
                global_step_val = tf.train.global_step(sess, global_step)
                
                feed_dict = {sat: batch_sat, grd: batch_grd, gt: batch_gt,
                                     learning_rate: learning_rate_val,
                                     keep_prob: keep_prob_val, training: True}
                _, loss_val, summary_val = sess.run([train_step, loss, summary], feed_dict=feed_dict)
                
                # Write to tensorboard
                writer.add_summary(summary_val, global_step_val)
                
                if iter % 200 == 0:
                    print('global %d, epoch %d, iter %d: loss : %.8f' % 
                          (global_step_val, epoch, iter, loss_val))
                    
                iter += 1
                
            # Save model
            model_dir = save_model_path+label+'/' + str(epoch) + '/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)

#             ---------------------- validation ----------------------

            print('validate...')
            print('go through all the ground images in the validation set...')
            input_data.reset_scan()
            distance = []
            while True:
                batch_sat, batch_grd, batch_gt = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break
                feed_dict = {sat: batch_sat, grd: batch_grd, gt: batch_gt, keep_prob: 1.0, training: False}
                heatmap_val = sess.run(heatmap, feed_dict=feed_dict)
                for batch_idx in range(batch_gt.shape[0]):
                    current_gt = batch_gt[batch_idx, :, :, :]
                    loc_gt = np.unravel_index(current_gt.argmax(), current_gt.shape)
                    current_pred = heatmap_val[batch_idx, :, :, :]
                    loc_pred = np.unravel_index(current_pred.argmax(), current_pred.shape)
                    distance.append(np.sqrt((loc_gt[0]-loc_pred[0])**2+(loc_gt[1]-loc_pred[1])**2)) 

            distance_error = np.mean(distance)
            print('mean distance error on validation set: ', distance_error)
            file = 'results/'+label+'_error.txt'
            with open(file,'ab') as f:
                np.savetxt(f, [distance_error], fmt='%4f', header='validation_set_mean_distance_error_in_pixels:', comments=str(epoch)+'_')

tf.reset_default_graph()
train(0)