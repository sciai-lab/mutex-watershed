import numpy as np
import torch

import mutex_watershed as mws
from neurofire.metrics.arand import ArandFromSegmentationBase


def compute_mws_segmentation(input_, offsets, n_attractive_channels,
                             strides, randomize_strides):
    vol_shape = input_.shape[1:]
    mst = mws.MutexWatershed(np.array(vol_shape), offsets,
                             n_attractive_channels, strides)
    if randomize_strides:
        mst.compute_randomized_bounds()
    sorted_edges = np.argsort(input_.ravel())
    mst.repulsive_mst_cut(sorted_edges) #, 0)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation


class ArandErrorFromMWS(ArandFromSegmentationBase):
    def __init__(self, offsets, strides=None, randomize_strides=False,
                 **super_kwargs):
        super().__init__(**super_kwargs)
        self.offsets = offsets
        self.dim = len(offsets[0])
        self.strides = strides
        self.randomize_strides = randomize_strides

    def _run_mws(self, input_):
        input_[self.dim:] *= -1
        input_[self.dim:] += 1
        return compute_mws_segmentation(input_, self.offsets, self.dim,
                                        strides=self.strides,
                                        randomize_strides=self.randomize_strides)

    def input_to_segmentation(self, input_batch):
        dim = input_batch.ndim - 2
        assert dim == self.dim
        seg = np.array([self._run_mws(batch) for batch in input_batch])
        # NOTE: we add a singleton channel axis here, which is expected by the arand metrics
        return torch.from_numpy(seg[:, None].astype('int32'))
