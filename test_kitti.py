import argparse
import importlib
import models
import numpy as np
import os
import tensorflow as tf
import time
import csv
from io_util import read_pcd, save_pcd
from visu_util import plot_pcd_three_views

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
def test(args):
    inputs = tf.placeholder(tf.float32, (1, None, 3))
    npts = tf.placeholder(tf.int32, (1,))
    gt = tf.placeholder(tf.float32, (1, args.num_gt_points, 3))
    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, npts, gt, tf.constant(1.0), args.is_training)

    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'completions'), exist_ok=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver()
    saver.restore(sess, args.checkpoint)
    csv_file = open(os.path.join(args.results_dir, 'results.csv'), 'w')
    writer = csv.writer(csv_file)
    writer.writerow(['id', 'total vehicles', 'total time', 'total points'])
    # pcd_file = os.listdir(args.pcd_dir)

    car_ids = [filename.split('.')[0] for filename in sorted(os.listdir(args.pcd_dir))]

    # frame_ids = [filename.split('_')[1] for filename in sorted(os.listdir(args.pcd_dir))]
    # print (frame_ids)
    # exit ()

    for j in range(50):
        count = 0
        total_time = 0
        total_points = 0
        for i, car_id in enumerate(car_ids):
            if car_id.split('_')[1] == str(j):
                partial = read_pcd(os.path.join(args.pcd_dir, '%s.pcd' % car_id))
                bbox = np.loadtxt(os.path.join(args.bbox_dir, '%s.txt' % car_id))
                # print (i)
                # print (car_id)
                # if i > 5:
                #     break
                # exit()
                total_points += partial.shape[0]

                # Calculate center, rotation and scale
                center = (bbox.min(0) + bbox.max(0)) / 2
                bbox -= center
                yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
                rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                    [np.sin(yaw), np.cos(yaw), 0],
                                    [0, 0, 1]])
                bbox = np.dot(bbox, rotation)
                scale = bbox[3, 0] - bbox[0, 0]
                bbox /= scale

                partial = np.dot(partial - center, rotation) / scale
                partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

                start = time.time()
                completion = sess.run(model.outputs, feed_dict={inputs: [partial], npts: [partial.shape[0]]})
                total_time += time.time() - start
                print (total_time)
                # exit()
                completion = completion[0]

                completion_w = np.dot(completion, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                completion_w = np.dot(completion_w * scale, rotation.T) + center
                pcd_path = os.path.join(args.results_dir, 'completions', '%s.pcd' % car_id)
                save_pcd(pcd_path, completion_w)
                count = count + 1
        # print('Average # input points:', total_points / len(car_ids))
        # print('Average time:', total_time / len(car_ids))                
        print('Average # input points:', total_points)
        print('Average time:', total_time)
        writer.writerow([j, count, total_time, total_points])
        # if i % args.plot_freq == 0:
        #     plot_path = os.path.join(args.results_dir, 'plots', '%s.png' % car_id)
        #     plot_pcd_three_views(plot_path, [partial, completion], ['input', 'output'],
        #                          '%d input points' % partial.shape[0], [5, 0.5])
    # print('Average # input points:', total_points / len(car_ids))
    # print('Average time:', total_time / len(car_ids))
    csv_file.close()
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='vpc_withoutfps')
    parser.add_argument('--checkpoint', default='log/vpc_withoutfps/model-100000')
    parser.add_argument('--pcd_dir', default='data/kitti/cars')
    parser.add_argument('--bbox_dir', default='data/kitti/bboxes')
    parser.add_argument('--results_dir', default='results/kitti_vpc_withoutfps')
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--plot_freq', type=int, default=100)
    parser.add_argument('--save_pcd', action='store_true')
    parser.add_argument('--is_training', type=bool, default=False)
    args = parser.parse_args()

    test(args)
