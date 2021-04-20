# Author: YAN  XIA 17/04/2020

import tensorflow as tf
from tf_util import *
from transform_nets import input_transform_net, feature_transform_net
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
# import transform_nets
class Model:
    def __init__(self, inputs, npts, gt, alpha, is_training):
        self.num_coarse = 1024
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_fine = self.grid_size ** 2 * self.num_coarse
        self.features = self.create_encoder(inputs, npts, is_training)
        self.coarse, self.fine, self.fine_new = self.create_decoder(self.features, is_training)
        self.loss, self.update = self.create_loss(self.coarse, self.fine, self.fine_new, gt, alpha)
        self.outputs = self.fine_new
        self.visualize_ops = [tf.split(inputs[0], npts, axis=0), self.coarse, self.fine, self.fine_new, gt]
        self.visualize_titles = ['input', 'coarse output', 'fine output', 'fine_new output', 'ground truth']

    def create_encoder(self, inputs, npts, is_training):
        b = tf.constant(is_training)
        with tf.variable_scope('transform_net1') as sc:
            transform = input_transform_net(inputs, b, bn_decay=None, K=3)
        point_cloud_transformed = tf.matmul(inputs, transform)

        with tf.variable_scope('encoder_0', reuse=tf.AUTO_REUSE):
            features = mlp_conv(inputs, [128, 256])
            features_global_1 = point_maxpool(features, npts)
            features_global = point_unpool(point_maxpool(features, npts, keepdims=True), npts)
            features = tf.concat([features, features_global], axis=2)

        with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE):
            features = mlp_conv(features, [512, 1024])
            features = point_maxpool(features, npts)
            global_feature = tf.concat([features_global_1, features], axis = 1)
        return global_feature

    def create_decoder(self, features, is_training):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            coarse = mlp(features, [1024, 1024, self.num_coarse * 3])
            coarse = tf.reshape(coarse, [-1, self.num_coarse, 3])

        with tf.variable_scope('folding', reuse=tf.AUTO_REUSE):
            x = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            y = tf.linspace(-self.grid_scale, self.grid_scale, self.grid_size)
            grid = tf.meshgrid(x, y)
            grid = tf.expand_dims(tf.reshape(tf.stack(grid, axis=2), [-1, 2]), 0)
            grid_feat = tf.tile(grid, [features.shape[0], self.num_coarse, 1])

            point_feat = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            point_feat = tf.reshape(point_feat, [-1, self.num_fine, 3])

            global_feat = tf.tile(tf.expand_dims(features, 1), [1, self.num_fine, 1])

            feat = tf.concat([grid_feat, point_feat, global_feat], axis=2)

            center = tf.tile(tf.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
            center = tf.reshape(center, [-1, self.num_fine, 3])

            fine = mlp_conv(feat, [512, 512, 3]) + center

            fine_features_1 = mlp_conv_single_bn(fine, [64], is_training)
            fine_features_global = point_maxpool_2(fine_features_1, keepdims=True)
            fine_features_global = point_unpool_2(fine_features_global,fine.shape[1])
            fine_features_2 = mlp_conv_single_bn(fine_features_1, [128], is_training)
            # fine_features_3 = mlp_conv_single_bn(fine_features_2, [256], is_training)
            fine_features_3 = mlp_conv_single_bn(fine_features_2, [1024], is_training)
            fine_features_4 = tf.concat([fine_features_3, fine_features_global], axis=2)
            fine_features_5 = mlp_conv_single_bn(fine_features_4, [512], is_training)
            fine_features_6 = mlp_conv_single_bn(fine_features_5, [256], is_training)
            fine_features_7 = mlp_conv_single_bn(fine_features_6, [128], is_training)
            fine_features_8 = mlp_conv_single_bn(fine_features_7, [3], is_training)
            fine_features_9 = tf.tanh(fine_features_8)

            fine_new = fine + fine_features_9

        return coarse, fine, fine_new

    def create_loss(self, coarse, fine, fine_new, gt, alpha):
        gt_ds = gt[:, :coarse.shape[1], :]
        loss_coarse = earth_mover(coarse, gt_ds)
        add_train_summary('train/coarse_loss', loss_coarse)
        update_coarse = add_valid_summary('valid/coarse_loss', loss_coarse)

        loss_fine = chamfer(fine, gt)
        add_train_summary('train/fine_loss', loss_fine)
        update_fine = add_valid_summary('valid/fine_loss', loss_fine)

        loss_fine_new = chamfer(fine_new, gt)
        add_train_summary('train/fine_new_loss', loss_fine_new)
        update_fine_new = add_valid_summary('valid/fine_new_loss', loss_fine_new)

        loss = loss_coarse + alpha * loss_fine + alpha * loss_fine_new
        add_train_summary('train/loss', loss)
        update_loss = add_valid_summary('valid/loss', loss)

        return loss, [update_coarse, update_fine, update_fine_new, update_loss]
