import h5py


def make_train_file():
    path = '/home/papec/Work/neurodata_hdd/isbi_2012/isbi2012_train_volume.h5'
    with h5py.File(path) as f:
        raw = f['volumes/raw'][:]
        gt = f['volumes/labels/neuron_ids_3d'][:]
        membranes = f['volumes/labels/membranes'][:]

    aff_path = '/home/papec/Work/neurodata_hdd/isbi_2012/predictions/isbi_train_offsetsV4_3d_meantda_damws2deval_final.h5'
    with h5py.File(aff_path) as f:
        affs = f['data'][:]

    with h5py.File('isbi_train_volume.h5') as f:
        f.create_dataset('raw', compression='gzip', data=raw, chunks=(1, 512, 512))
        f.create_dataset('labels/gt_segmentation', compression='gzip', data=gt, chunks=(1, 512, 512))
        f.create_dataset('labels/membrabes', compression='gzip', data=membranes, chunks=(1, 512, 512))
        f.create_dataset('affinities', compression='gzip', data=affs, chunks=(3, 1, 512, 512))


def make_test_file():
    path = '/home/papec/Work/neurodata_hdd/isbi_2012/isbi2012_test_volume.h5'
    with h5py.File(path) as f:
        raw = f['volumes/raw'][:]

    aff_path = '/home/papec/Work/neurodata_hdd/isbi_2012/predictions/isbi_test_offsetsV4_3d_meantda_damws2deval_final.h5'
    with h5py.File(aff_path) as f:
        affs = f['data'][:]

    with h5py.File('isbi_test_volume.h5') as f:
        f.create_dataset('raw', compression='gzip', data=raw, chunks=(1, 512, 512))
        f.create_dataset('affinities', compression='gzip', data=affs, chunks=(3, 1, 512, 512))


if __name__ == '__main__':
    make_test_file()
