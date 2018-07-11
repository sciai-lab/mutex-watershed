import numpy as np
import h5py
from scipy.ndimage import convolve
from cremi_tools.viewer.volumina import view


def make_2d_edges(segmentation):
    gx = convolve(segmentation + 1, np.array([-1., 1.]).reshape(1, 2))
    gy = convolve(segmentation + 1, np.array([-1., 1.]).reshape(2, 1))
    return ((gx ** 2 + gy ** 2) > 0)


def make_edges(segmentation):
    edges = np.zeros_like(segmentation, dtype='uint64')
    for z in range(segmentation.shape[0]):
        edges[z] = make_2d_edges(segmentation[z])
    return edges


def view_res():
    raw_path = '/home/cpape/Work/data/isbi2012/isbi2012_test_volume.h5'
    with h5py.File(raw_path) as f:
        raw = f['volumes/raw'][:]

    seg_path = './results/mws.h5'
    with h5py.File(seg_path) as f:
        seg = f['data'][:]

    edges = make_edges(seg)

    view([raw, seg, edges], ['raw', 'mws', 'edges'])


if __name__ == '__main__':
    view_res()
