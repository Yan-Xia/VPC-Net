# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import argparse
import os
from io_util import read_pcd
from tensorpack import DataFlow, dataflow


class pcd_df(DataFlow):
    def __init__(self, model_list, num_scans, partial_dir, complete_dir):
        self.model_list = model_list
        self.num_scans = num_scans
        self.partial_dir = partial_dir
        self.complete_dir = complete_dir

    def size(self):
        return len(self.model_list) * self.num_scans

    def get_data(self):
        for model_id in model_list:
            #print (os.path.join(self.complete_dir, '%s.pcd' % model_id))
            #exit()
            complete = read_pcd(os.path.join(self.complete_dir, '%s.pcd' % model_id))
            #for i in range(self.num_scans):
            #    partial = read_pcd(os.path.join(self.partial_dir, model_id, '%d.pcd' % i))
            #    yield model_id.replace('/', '_'), partial, complete
            partial = read_pcd(os.path.join(self.partial_dir, '%s.pcd' % model_id))
            yield model_id.replace('/', '_'), partial, complete

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_path')
    parser.add_argument('--num_scans', type=int)
    parser.add_argument('--partial_dir')
    parser.add_argument('--complete_dir')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    with open(args.list_path) as file:
        model_list = file.read().splitlines()
    df = pcd_df(model_list, args.num_scans, args.partial_dir, args.complete_dir)
    if os.path.exists(args.output_path):
        os.system('rm %s' % args.output_path)
    dataflow.LMDBSerializer.save(df, args.output_path)
