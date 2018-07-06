from concurrent import futures
import numpy as np
import mutex_watershed as mws


def run_mws(affinities,
            offsets, stride,
            seperating_channel=2,
            invert_dam_channels=True,
            bias_cut=0.,
            randomize_bounds=True):
    assert len(affinities) == len(offsets), "%s, %i" % (str(affinities.shape), len(offsets))
    affinities_ = np.require(affinities.copy(), requirements='C')
    if invert_dam_channels:
        affinities_[seperating_channel:] *= -1
        affinities_[seperating_channel:] += 1
    affinities_[:seperating_channel] += bias_cut
    sorted_edges = np.argsort(affinities_.ravel())
    # run the mst watershed
    vol_shape = affinities_.shape[1:]
    mst = mws.MutexWatershed(np.array(vol_shape),
                             offsets,
                             seperating_channel,
                             stride)
    if randomize_bounds:
        mst.compute_randomized_bounds()
    mst.repulsive_ucc_mst_cut(sorted_edges, 0)
    segmentation = mst.get_flat_label_image().reshape(vol_shape)
    return segmentation


def stacked_mst(affinities,
                offsets,
                stride,
                seperating_channel=2,
                invert_dam_channels=True):
    out_shape = affinities.shape[1:]
    seg = np.zeros(out_shape, dtype='uint32')

    def run_mst_z(z):
        seg_z = run_mws(
            affinities[:, z, :], offsets, stride, seperating_channel, invert_dam_channels=invert_dam_channels
        )
        seg[z] = seg_z
        return seg_z.max() + 1

    with futures.ThreadPoolExecutor(max_workers=8) as tp:
        tasks = [tp.submit(run_mst_z, z) for z in range(len(seg))]
        offsets = [t.result() for t in tasks]

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets).astype('uint32')
    seg += offsets[:, None, None]

    return seg


def mst_2d_from_3d(affinities, offsets, invert_dam_channels=False):
    keep_channels = [i for i, off in enumerate(offsets) if off[0] == 0]
    offsets_ = [off[1:] for i, off in enumerate(offsets) if i in keep_channels]
    affinities_ = np.require(affinities[keep_channels], requirements='C')

    out_shape = affinities_.shape[1:]
    seg = np.zeros(out_shape, dtype='uint32')
    actions = np.zeros(out_shape, dtype='uint32')

    def mst2d(z):
        mst, action = run_mws(np.require(affinities_[:, z], requirements='C'),
                              offsets_,
                              np.array([2, 2]),
                              seperating_channel=2,
                              invert_dam_channels=invert_dam_channels)
        seg[z] = mst
        actions[z] = action
        return mst.max() + 1

    with futures.ThreadPoolExecutor(max_workers=8) as tp:
        tasks = [tp.submit(mst2d, z) for z in range(affinities_.shape[1])]
        offsets = [t.result() for t in tasks]

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets).astype('uint32')
    seg += offsets[:, None, None]

    return seg
