'''
nosetests picks up all python files that start or end with `test` in their name,
and runs all methods that start or end with `test`.
'''

import matplotlib
matplotlib.use('Agg')

# pythonpath modification to make learnedWS availabel
# for import without requiring it to be installed
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# import learnedWS
import mutex_watershed as mws
import numpy as np
import numpy.ma as ma

import matplotlib.pyplot as plt

import h5py

from vigra.analysis import regionImageToCrackEdgeImage
from vigra import gaussianSmoothing

from scipy.ndimage import convolve
from skimage.morphology import binary_dilation

import time


def get_tolerace_mask(target_labels, tollerance, image_shape):
    # compute_boundary_mask
    mask = np.zeros(image_shape)
    gx = convolve(target_labels + 1, np.array([-1., 0., 1.]).reshape(1, 3))
    gy = convolve(target_labels + 1, np.array([-1., 0., 1.]).reshape(3, 1))
    boundary_mask = ((gx ** 2 + gy ** 2) > 0)
    for m in range(1, tollerance):
        boundary_mask = binary_dilation(boundary_mask)
    return boundary_mask

def test_cuf():

    C_ATTRACTIVE = 3
    EDGE_LEN = 256
    Z_EDGE_LEN = 10
    USE_ALL = False

    offsets = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1],
                    [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
                    [-2, 0, 0], [0, -9, 0], [0, 0, -9],
                    [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
                    [-3, 0, 0], [0, -27, 0], [0, 0, -27]], dtype=np.int)


    with h5py.File("../../data/unet_for_damws_sampleA.h5", "r") as aff:
        if USE_ALL:
            affinities = aff["data"].value
        else:
            print("not actually large data :)")
            affinities = aff["data"][:, :Z_EDGE_LEN, :EDGE_LEN, :EDGE_LEN]
    print("data loading successfull")

    affinities[C_ATTRACTIVE:] *= -1
    affinities[C_ATTRACTIVE:] += 1.

    weight_shape = affinities.shape
    image_shape = weight_shape[1:]

    f_affinities = affinities.flatten()
    print("sorting...")
    mask = np.zeros(affinities.shape, dtype=np.int)
    mask[:, :, ::12, ::12] = 1
    masked_aff = ma.array(f_affinities, mask=mask, fill_value=999)
    sorted_edges = np.argsort(masked_aff)[:f_affinities.shape[0]//144]
    print("initializing...")
    start = time.time()
    MST_calculator = mws.MutexWatershed(np.array(image_shape), offsets, C_ATTRACTIVE, np.array([1,12,12]))
    end = time.time()
    print("init time ", end - start)

    # start = time.time()
    # MST_calculator.compute_randomized_bounds()
    # end = time.time()
    # print("randomize bounds...", end - start)

    start = time.time()
    print("calculating...")
    MST_calculator.repulsive_mst_cut(sorted_edges)
    end = time.time()
    print("time ", end - start)

    with h5py.File("large_label_image.h5", "w") as lab:
        lab.create_dataset("data", data=MST_calculator.get_flat_label_image().reshape(image_shape), compression='gzip')

if __name__ == '__main__':
    test_cuf()
