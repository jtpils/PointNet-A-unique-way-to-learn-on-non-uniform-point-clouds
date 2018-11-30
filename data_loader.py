# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import h5py

BASE_DIR = os.path.dirname(os.path.abspath('/home/rahulchakwate/My_tensorflow/3D Object Segmentation/PointNet Implementation/data'))
sys.path.append(BASE_DIR)


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

TRAIN_FILES = getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES=getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def loadDataFile(filename):
    return load_h5(filename)

LOG_FOUT = open(os.path.join(BASE_DIR, 'log_train.txt'), 'w')

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def load_3D_data(mode):
    if mode=='train' :
     FILES=TRAIN_FILES
    elif mode=='test':
     FILES=TEST_FILES
     
    file_idxs = np.arange(0, len(FILES))
    np.random.shuffle(file_idxs)

    NUM_POINT=1024
    fn=0
    log_string('----' + str(fn) + '-----')
    current_data, current_label = loadDataFile(FILES[file_idxs[fn]])
    current_data = current_data[:,0:NUM_POINT,:]
    #current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
    current_label = np.squeeze(current_label)
    d0=current_data
    l0=current_label

    for fn in range(1,len(FILES)):
        
        log_string('----' + str(fn) + '-----')
        current_data, current_label = loadDataFile(FILES[file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        #current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))            
        current_label = np.squeeze(current_label)
        d1=current_data
        l1=current_label
        d0=np.concatenate((d0,d1),axis=0)
        l0=np.concatenate((l0,l1))

    return d0,l0

