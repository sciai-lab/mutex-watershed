import os
import time
import argparse
import numpy as np
import h5py

from scipy.ndimage import convolve


def writeHDF5(data, path, key, **kwargs):
    with h5py.File(path) as f:
        f.create_dataset(key, data=data, **kwargs)


def make_2d_edges(segmentation):
    gx = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(1, 3, 1))
    gy = convolve(segmentation + 1, np.array([-1., 0., 1.]).reshape(3, 1, 1))
    return ((gx ** 2 + gy ** 2) > 0)


def make_pmap(affinities):
    return np.maximum(affinities[1], affinities[2])


def make_pmap3d(affinities):
    return np.mean(affinities[:3])


def timer(func):
    def func_wrapper(*args, **kwargs):
        t0 = time.time()
        segmentation = func(*args, **kwargs)
        return segmentation, time.time() - t0
    return func_wrapper


@timer
def threshold_baseline(pmap, threshold):
    import vigra
    pmap_segmentation = np.zeros_like(pmap, dtype='uint32')
    pmap_segmentation[pmap > threshold] = 1
    for z in range(pmap_segmentation.shape[0]):
        pmap_segmentation[z] = vigra.analysis.labelImageWithBackground(pmap_segmentation[z])
    return pmap_segmentation


def compute_wsdt(input_, threshold_dt, sigma_seeds, size_filter,
                 is_anisotropic=False):
    from cremi_tools.segmentation.watershed import DTWatershed
    segmenter = DTWatershed(threshold_dt, sigma_seeds, size_filter=size_filter,
                            is_anisotropic=is_anisotropic, n_threads=1)
    return segmenter(input_)[0]


def compute_ws(input_, sigma, size_filter,
               is_anisotropic=False):
    from cremi_tools.segmentation.watershed import Watershed
    segmenter = Watershed(sigma, size_filter, is_anisotropic=is_anisotropic,
                          n_threadds=1)
    return segmenter(input_)[0]


@timer
def wsdt_baseline(pmap, threshold, sigma, min_seg_size):
    return compute_wsdt(pmap, threshold, sigma, min_seg_size,
                        is_anisotropic=True)


@timer
def ws_baseline(pmap, sigma, min_seg_size):
    return compute_ws(pmap, sigma, min_seg_size,
                      is_anisotropic=True)


@timer
def ws3d_baseline(pmap, sigma, min_seg_size):
    sigma = (sigma / 10, sigma, sigma)
    return compute_ws(pmap, sigma, min_seg_size)


@timer
def wsdt3d_baseline(pmap, threshold, sigma, min_seg_size):
    sigma = (sigma / 10, sigma, sigma)
    return compute_wsdt(pmap, threshold, sigma, min_seg_size,
                        is_anisotropic=False)


# TODO
@timer
def mc_baseline(affinities):
    from mc_baselines import compute_mc_superpixels
    return compute_mc_superpixels(affinities, n_threads=8)


@timer
def mc_longrange_baseline(affinities, offsets, only_repulsive=False):
    from mc_baselines import compute_long_range_mc_superpixels
    return compute_long_range_mc_superpixels(affinities, offsets,
                                             only_repulsive_lr=only_repulsive,
                                             n_threads=8)


def lmc_baseline():
    pass


@timer
def mws_result(affinities, offsets, strides, randomize_bounds):
    from run_mws import run_mws
    return run_mws(affinities, offsets, strides,
                   randomize_bounds=randomize_bounds, seperating_channel=3)


# view affinities and / or segmentations
def view_results(raw_path,
                 raw_key,
                 pmaps,
                 segmentations,
                 segmentation_labels):
    from cremi_toos.viewer.volumina import view
    with h5py.File(raw_path) as f:
        raw = f[raw_key][:].astype('float32')
    data = [raw]
    data.extend(pmaps)
    labels = ['raw']
    labels.extend(['pmap-%i' % i for i in range(len(pmaps))])

    for seg in segmentations:
        edges = make_2d_edges(seg)
        data.extend([seg, edges])
    for seg_label in segmentation_labels:
        labels.append(seg_label)
        labels.append(seg_label + "_edges")
    view(data, labels)


def isbi_experiments(raw_path, aff_path,
                     raw_key, aff_key,
                     result_folder,
                     threshold=False,
                     ws=False,
                     wsdt=False,
                     mc=False,
                     mc_lr=False,
                     mc_lr_repulsive=False,
                     mws=False,
                     view=False):
    # affinity offsets
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               # direct 3d nhood for attractive edges
               [-1, -1, -1], [-1, 1, 1], [-1, -1, 1], [-1, 1, -1],
               # indirect 3d nhood for dam edges
               [0, -9, 0], [0, 0, -9],
               # long range direct hood
               [0, -9, -9], [0, 9, -9], [0, -9, -4], [0, -4, -9], [0, 4, -9], [0, 9, -4],
               # inplane diagonal dam edges
               [0, -27, 0], [0, 0, -27]]
    # additional long range dam edges

    with h5py.File(aff_path) as f:
        affs = f[aff_key][:]
    pmap = make_pmap(affs)

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    segmentations = []
    labels = []
    if threshold:
        thresh = 0.9
        print("Computing threshold segmentation ...")
        thresh_seg, t_thresh = threshold_baseline(1. - pmap, thresh)
        print("... finished in %f s" % t_thresh)
        writeHDF5(thresh_seg, os.path.join(result_folder, 'threshold.h5'),
                  'data', compression='gzip')
        segmentations.append(thresh_seg)
        labels.append('thresholded')

    if ws:
        min_seg_size = 25
        sigma_ws = 1.6
        print("Computing watershed segmentation ...")
        ws_seg, t_ws = ws_baseline(pmap, sigma_ws, min_seg_size)
        print("... finished in  %f s" % t_ws)
        writeHDF5(ws_seg, os.path.join(result_folder, 'ws.h5'),
                  'data', compression='gzip')
        segmentations.append(ws_seg)
        labels.append('watershed')

    if wsdt:
        sigma_wsdt = 3.
        threshold = 0.25
        min_seg_size = 25
        print("Computing distance transform watershed segmentation ...")
        wsdt_seg, t_wsdt = wsdt_baseline(pmap, threshold, sigma_wsdt, min_seg_size)
        print("... finished in  %f s" % t_wsdt)
        writeHDF5(wsdt_seg, os.path.join(result_folder, 'wsdt.h5'),
                  'data', compression='gzip')
        segmentations.append(wsdt_seg)
        labels.append('watershed on distance transform')

    if mc:
        print("Computing multicut segmentation ...")
        mc_seg, t_mc = mc_baseline(affs)
        print("... finished in  %f s" % t_mc)
        writeHDF5(mc_seg, os.path.join(result_folder, 'mc.h5'),
                  'data', compression='gzip')
        segmentations.append(mc_seg)
        labels.append('local multicut')

    if mc_lr:
        print("Computing multicut segmentation with long-range edges ...")
        mc_lr_seg, t_mclr = mc_longrange_baseline(affs, offsets)
        print("... finished in %f s" % t_mclr)
        writeHDF5(mc_lr_seg, os.path.join(result_folder, 'mclr.h5'),
                  'data', compression='gzip')
        segmentations.append(mc_lr_seg)
        labels.append('long range multicut with all edges')

    if mc_lr_repulsive:
        print("Computing multicut segmentation with repulsive long-range edges ...")
        mc_lr_seg2, t_mclr2 = mc_longrange_baseline(affs, offsets, True)
        print("... finished in %f s" % t_mclr2)
        writeHDF5(mc_lr_seg2, os.path.join(result_folder, 'mclr-repulsive.h5'),
                  'data', compression='gzip')
        segmentations.append(mc_lr_seg2)
        labels.append('long range multicut only repulsive edges')

    if mws:
        strides = np.array([1., 10., 10.])
        print("Computing mutex watershed segmentation ...")
        mws_seg, t_mws = mws_result(affs, offsets, strides, randomize_bounds=False)
        print("... finished in  %f s" % t_mws)
        writeHDF5(mws_seg, os.path.join(result_folder, 'mws.h5'),
                  'data', compression='gzip')
        segmentations.append(mws)
        labels.append('MWS')

    pmaps = [pmap]
    if view:
        view_results(raw_path, raw_key, pmaps, segmentations, labels)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def main():
    algos = ['threshold', 'ws', 'wsdt', 'mc', 'mclr', 'mclr-repulsive', 'mws']
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_path', type=str, help='path to raw data (hdf5 file)')
    parser.add_argument('raw_key', type=str, help='path to raw dataset in hdf5 file')
    parser.add_argument('aff_path', type=str, help='path to affinities (hdf5 file)')
    parser.add_argument('aff_key', type=str, help='path to affinity dataset in hdf5 file')
    parser.add_argument('result_folder', type=str,
                        help='folder to save result segmentations as hdf5')
    parser.add_argument('--algorithms', type=str, nargs='+', default=['mws'],
                        help='list of algorithms to evaluate, possiblve:' + ', '.join(algos))
    parser.add_argument('--view', type=str2bool, default='n',
                        help='open viewer with all results (needs volumina)')

    args = parser.parse_args()
    raw_path, aff_path = args.raw_path, args.aff_path
    raw_key, aff_key = args.raw_key, args.aff_key
    assert os.path.exists(raw_path), raw_path
    assert os.path.exists(aff_path), aff_path
    algo_choice = args.algorithms
    assert all(alg in algos
               for alg in algo_choice), "Invalid algorithm choice in:" + ", ".join(algo_choice)
    isbi_experiments(raw_path, aff_path,
                     raw_key, aff_key, args.result_folder,
                     threshold='threshold' in algo_choice,
                     ws='ws' in algo_choice,
                     wsdt='wsdt' in algo_choice,
                     mc='mc' in algo_choice,
                     mc_lr='mclr' in algo_choice,
                     mc_lr_repulsive='mlr-repulsive' in algo_choice,
                     mws='mws' in algo_choice,
                     view=args.view)


if __name__ == '__main__':
    main()
