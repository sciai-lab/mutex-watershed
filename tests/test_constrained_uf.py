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

    C_ATTRACTIVE = 4
    EDGE_LEN = 512

    out_label = np.empty((30, EDGE_LEN, EDGE_LEN), dtype=np.int64)

    if C_ATTRACTIVE == 2:
        offsets = np.array([[-1,0],[0,-1],[-27,0],[0,-27]], dtype=np.int)
    elif C_ATTRACTIVE == 3:
        offsets = np.array([[-1,0],[0,-1],[-1,-1],[-27,0],[0,-27],[-27,-27]], dtype=np.int)
    else:
        offsets = np.array([[-1,0],[0,-1],[-1,-1],[+1,-1],[-27,0],[0,-27],[-27,-27],[27,-27]], dtype=np.int)

    for z_slice in [15]:

        with h5py.File("/mnt/localdata01/swolf/ws/isbi/isbi_for_steffen/isbi_test_100k.h5", "r") as aff:
            if C_ATTRACTIVE == 2:
                affinities = aff["data"][:, z_slice, :EDGE_LEN, :EDGE_LEN][[0,1,12,13]]
            elif C_ATTRACTIVE == 3:
                affinities = aff["data"][:, z_slice, :EDGE_LEN, :EDGE_LEN][[0,1,2,12,13,14]]
            else:
                affinities = aff["data"][:, z_slice, :EDGE_LEN, :EDGE_LEN][[0,1,2,3,12,13,14,15]]

        affinities[C_ATTRACTIVE:] *= -1
        affinities[C_ATTRACTIVE:] += 1.

        weight_shape = affinities.shape
        image_shape = weight_shape[1:]

        label_image = np.zeros(image_shape, dtype=np.int)
        label_image[:EDGE_LEN//2] = 1
        label_image[:, :EDGE_LEN//2] += 2
        label_image += 10

        # affinities[:, :, C_ATTRACTIVE:] += affinities[:, :, :C_ATTRACTIVE]
        # affinities[:, :, C_ATTRACTIVE:] /= 5

        # affinities[:, :, C_ATTRACTIVE:] += 10.
        # affinities[::4 ,::4, C_ATTRACTIVE:] -= 10.
        # for k in range(affinities.shape[0]):
        #     bound = np.array([MST_calculator._check_bounds(i, k) for i in range(label_image.flatten().size)])
        #     plt.imshow(bound.reshape(image_shape), interpolation=None)
        #     plt.savefig('bound_{}.png'.format(k), dpi=500)
        #     plt.close()
        f_affinities = affinities.flatten()

        for epoch in range(1):
            sorted_edges = np.argsort(f_affinities)
            MST_calculator = mws.MutexWatershed(np.array(image_shape), offsets, C_ATTRACTIVE, np.array([1,1,1]))
            # MST_calculator.clear_all()
            MST_calculator.set_gt_labels(label_image.flatten())

            start = time.time()
            gradients = MST_calculator.get_gradients(sorted_edges, f_affinities, 0)
            # MST_calculator.repulsive_ucc_mst_cut(sorted_edges, 0)#[:int(EDGE_LEN*EDGE_LEN*C_ATTRACTIVE*0.5)])
            end = time.time()

            print("time ", end - start)
            # # sorted_edges = np.argsort(f_affinities)
            # MST_calculator.clear_all()
            # MST_calculator.set_gt_labels(label_image.flatten())

            # start = time.time()
            # # MST_calculator.get_gradients(sorted_edges, f_affinities, 0)
            # MST_calculator.repulsive_ucc_mst_cut(sorted_edges, 0)#[:int(EDGE_LEN*EDGE_LEN*C_ATTRACTIVE*0.5)])
            # end = time.time()


            # # errors_edges = MST_calculator.get_smoothmax_gradients2(f_affinities, -1)
            # print("time ", end - start)

            # # error_image = np.zeros(sorted_edges.shape)
            # # for k, e in errors_edges.items():
            # #     for x in e:
            # #         error_image[x] = k[2]

            # # error_image = error_image.reshape(affinities.shape)


            # # f_affinities -= 1000 *(errors_edges)
            # # f_affinities = np.clip(f_affinities,0,1)


            for k in range(affinities.shape[0]):
                plt.imshow(np.sign(gradients).reshape(affinities.shape)[k], interpolation=None)
                plt.savefig('new_grads_{}_{}.png'.format(epoch, k), dpi=500)
                plt.close()

            # for k in range(affinities.shape[0]):
            #     plt.imshow(errors_edges.reshape(affinities.shape)[k], interpolation=None)
            #     plt.savefig('error_edges_{}_{}.png'.format(epoch, k), dpi=500)
            #     plt.close()


        #     fin_labels = MST_calculator.get_flat_label_image().reshape(image_shape)
        #     fin_labels += ((fin_labels.max()) * (fin_labels % 10))

        #     out_label[z_slice] = MST_calculator.get_flat_label_image().reshape(image_shape)
        #     plt.imshow(regionImageToCrackEdgeImage(fin_labels.astype(np.uint32)), interpolation=None)
        #     plt.savefig('seg_{}.png'.format(epoch), dpi=500)
        #     plt.close()

        # for k in range(affinities.shape[0]):
        #     plti.mshow(f_affinities.reshape(affinities.shape)[k], interpolation=None)
        #     plt.savefig('aff_{}_{}.png'.format(epoch, k), dpi=500)
        #     plt.close()

        exit()


        plt.imshow(regionImageToCrackEdgeImage(label_image.astype(np.uint32)), interpolation=None)
        plt.savefig('gt_seg.png', dpi=500)
        plt.close()

        # out_label[z_slice] = MST_calculator.get_flat_label_image().reshape(image_shape)
        plt.imshow(regionImageToCrackEdgeImage(MST_calculator.get_flat_c_label_image().reshape(image_shape).astype(np.uint32)), interpolation=None)
        plt.savefig('c_seg.png', dpi=500)
        plt.close()

        acts = MST_calculator.get_flat_applied_uc_actions().reshape((-1, EDGE_LEN, EDGE_LEN))
        for k in range(acts.shape[0]):
            plt.imshow(acts[k], interpolation=None)
            plt.savefig('a_{}.png'.format(k), dpi=500)
            plt.close()

        acts = MST_calculator.get_flat_applied_c_actions().reshape((-1, EDGE_LEN, EDGE_LEN))
        for k in range(acts.shape[0]):
            plt.imshow(acts[k], interpolation=None)
            plt.savefig('ca_{}.png'.format(k), dpi=500)
            plt.close()


        for k in range(affinities.shape[0]):
            plt.imshow(affinities[k], interpolation=None)
            plt.savefig('aff_{}.png'.format(k), dpi=500)
            plt.close()


    with h5py.File("out.h5", "w") as out:
        out.create_dataset("data", data=out_label, compression='gzip')


if __name__ == '__main__':
    test_cuf()
