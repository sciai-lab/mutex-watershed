import sys
import h5py
sys.path.append('../isbi')
from mc_baselines import compute_long_range_mc_superpixels, compute_lmc_superpixels


# def compute_long_range_mc_superpixels(affinities, offsets,
#                                       only_repulsive_lr, n_threads):
#     segmenter = LongRangeMulticutSuperpixel(only_repulsive_lr=only_repulsive_lr,
#                                             stacked_2d=True, n_threads=n_threads)
#     return segmenter(affinities)


def run_mclr(affs, offsets, only_replsive=False):
    seg = compute_long_range_mc_superpixels(affs, offsets, only_replsive, n_threads=1,
                                            stacked_2d=False)
    return seg


def run_lmc(affs, offsets):
    seg = compute_lmc_superpixels(affs, offsets, n_threads=1, stacked_2d=False)
    return seg


if __name__ == '__main__':
    input_file = '/home/cpape/Downloads/im_155.h5'
    with h5py.File(input_file) as f:
        affs = f['data'][:]

    offsets = [[-1, 0, 0], [0, -1, 0],
               [-9, 0], [0, -9],
               [-9, -9], [9, -9], [-9, -4], [-4, -9], [4, -9], [9, -4],
               [-27, 0], [0, -27]]

    # long range multiuct
    seg = run_mclr(affs, offsets)
    print(seg.shape)
    with h5py.File('seg_mc.h5') as f:
        f.create_dataset('data', data=seg, compression='gzip')

    # lifted multicut
    # seg = run_lmc(affs, offsets)
    # print(seg.shape)
    # with h5py.File('seg_lmc.h5') as f:
    #     f.create_dataset('data', data=seg, compression='gzip')
