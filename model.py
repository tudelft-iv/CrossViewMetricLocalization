from VGG import VGG16
import tensorflow as tf
from output_head import decoder

# SAFA module from https://proceedings.neurips.cc/paper/2019/file/ba2f0015122a5955f8b3a50240fb91b2-Paper.pdf
def spatial_aware(input_feature, dimension, trainable, name, reuse):
    batch, height, width, channel = input_feature.get_shape().as_list()
    vec1 = tf.reshape(tf.reduce_mean(input_feature, axis=-1), [-1, height * width])
    with tf.variable_scope(name, reuse=reuse):
        weight1 = tf.get_variable(name='weights1', shape=[height * width, int(height * width/2), dimension],
                                 trainable=trainable,
                                 initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias1 = tf.get_variable(name='biases1', shape=[1, int(height * width/2), dimension],
                               trainable=trainable, initializer=tf.constant_initializer(0.1),
                               regularizer=tf.contrib.layers.l1_regularizer(0.01))

        vec2 = tf.einsum('bi, ijd -> bjd', vec1, weight1) + bias1

        weight2 = tf.get_variable(name='weights2', shape=[int(height * width / 2), height * width, dimension],
                                  trainable=trainable,
                                  initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.005),
                                  regularizer=tf.contrib.layers.l2_regularizer(0.01))
        bias2 = tf.get_variable(name='biases2', shape=[1, height * width, dimension],
                                trainable=trainable, initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l1_regularizer(0.01))
        vec3 = tf.einsum('bjd, jid -> bid', vec2, weight2) + bias2
        
        return vec3


def CVML(x_sat, x_grd, keep_prob, dimension, trainable, reuse):
    
    # grd
    vgg_grd = VGG16()
    grd_local, _, _, _, _ = vgg_grd.VGG16_conv(x_grd, keep_prob, trainable, 'VGG_grd', reuse)

    grd_local = tf.nn.max_pool(grd_local, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    batch, g_height, g_width, channel = grd_local.get_shape().as_list()

    grd_w = spatial_aware(grd_local, dimension, trainable, name='spatial_grd', reuse=reuse)
    grd_local = tf.reshape(grd_local, [-1, g_height * g_width, channel])

    grd_global = tf.einsum('bic, bid -> bdc', grd_local, grd_w)
    grd_global = tf.reshape(grd_global, [-1, 1, 1, dimension*channel])
    grd_global = tf.nn.l2_normalize(grd_global, dim=-1)

    # sat
    vgg_sat = VGG16()
    sat_local, sat512, sat256, sat128, sat64 = vgg_sat.VGG16_conv(x_sat, keep_prob, trainable, 'VGG_sat', reuse)
    
    _, s_height, s_width, channel = sat_local.get_shape().as_list()
    sat_split = 8 # split satellite feature into 8*8 sub-volumes
    
    for i in range(0, sat_split):
        strip_horizontal = sat_local[:, tf.cast(i*s_height/sat_split, tf.int32):tf.cast((i+1)*s_height/sat_split, tf.int32), :, :]
        sat_global_horizontal = []
        for j in range(0, sat_split):
            patch = strip_horizontal[:, :, tf.cast(j*s_height/sat_split, tf.int32):tf.cast((j+1)*s_height/sat_split, tf.int32), :]
            
            # Feed each satellite sub-volume into the SAFA module
            if i == 0 and j == 0:
                copy_weights = False
            else:
                copy_weights = True
            if reuse == True:
                copy_weights = True
            sat_w = spatial_aware(patch, dimension, trainable, name='spatial_sat', reuse=copy_weights)
            _, p_height, p_width, channel = patch.get_shape().as_list()
            patch = tf.reshape(patch, [-1, p_height * p_width, channel])
            
            patch_global =  tf.einsum('bic, bid -> bdc', patch, sat_w)
            patch_global = tf.reshape(patch_global, [-1, 1, 1, dimension*channel])
            patch_global = tf.nn.l2_normalize(patch_global, dim=-1)
            
            if j == 0:
                sat_global_horizontal = patch_global
            else:
                sat_global_horizontal = tf.concat([sat_global_horizontal, patch_global], 2)
        if i == 0:
            sat_global = sat_global_horizontal
        else:
            sat_global = tf.concat([sat_global, sat_global_horizontal], 1)
        
    grd_global_broadcasted = tf.broadcast_to(grd_global, [tf.shape(grd_global)[0], sat_split, sat_split, tf.shape(grd_global)[-1]])
    matching_score = tf.reduce_sum(tf.multiply(grd_global_broadcasted, sat_global), axis=-1, keepdims=True) # cosine similarity
    cost_map = tf.concat([matching_score, sat_global], 3)

    costmap_decoder = decoder()
    logits = costmap_decoder.decode(cost_map, keep_prob, trainable, 'decode', sat512, sat256, sat128, sat64, sat_local, reuse)

    return logits, matching_score

