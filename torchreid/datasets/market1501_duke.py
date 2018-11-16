from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np
import h5py
from scipy.misc import imsave

from .bases import BaseImageDataset

class Market1501_Duke(object):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html
    
    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'
    dataset_dir2 = 'dukemtmc-reid'

    def __init__(self, root='data', verbose=True, **kwargs):
        super(Market1501_Duke, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.dataset_dir2 = osp.join(root, self.dataset_dir2)

        ######
        self.train_dir2 = osp.join(self.dataset_dir2, 'DukeMTMC-reID/bounding_box_train')
        train2, num_train_pids2, num_train_imgs2 = self._process_dir2(self.train_dir2, relabel=True)
        ######

        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_imgs = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs = self._process_dir(self.gallery_dir, relabel=False)
        
        num_total_pids = num_train_pids + num_query_pids + num_train_pids2 -1 #backgroud-1
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs + num_train_imgs2
        num_train_pids = num_train_pids + num_train_pids2 -1
        num_train_imgs = num_train_imgs + num_train_imgs2

        if verbose:
            print("=> Market1501 loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # images")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
            print("  ------------------------------")

        self.train = train + train2
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids 
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.num_train_imgs     = num_train_imgs
        self.num_query_imgs     = num_query_imgs
        self.num_gallery_imgs   = num_gallery_imgs
        self.num_train_cams     = 14
        self.num_query_cams     = 6
        self.num_gallery_cams   = 6
        #self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        #self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        #self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        print(len(self.train))

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            #dataset_id = np.array(np.zeros(2),dtype=(np.float32))
            #dataset_id[:1]=1
            dataset_id=0
            dataset.append((img_path, pid, camid,dataset_id))

        num_pids = len(pid_container)
        num_imgs = len(dataset)


        return dataset, num_pids, num_imgs
    
    def _process_dir2(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}
 
        dataset = []

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1 # index starts from 0

            #
            camid += 6 #index starts from 6

            if relabel: pid = pid2label[pid]

            if pid != 0:
                pid = pid + 750

            #dataset_id = np.array(np.zeros(2),dtype=(np.float32))
            dataset_id=1
            dataset.append((img_path, pid, camid, dataset_id))

        num_pids = len(pid_container)
        num_imgs = len(dataset)

        return dataset, num_pids, num_imgs