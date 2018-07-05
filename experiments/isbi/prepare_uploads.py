import os
import argparse
import vigra
from isbi_experiments import make_2d_edges


def prepare_upload(seg_path, seg_key, out_path):
    # load the segmentation
    seg = vigra.readHDF5(seg_path, seg_key)
    # convert segmentation to edge volume
    edges = make_2d_edges(seg)
    # make tif for upload
    edges = 1 - edges.transpose((2, 1, 0))
    edges = edges.astype('float32')
    vigra.impex.writeVolume(edges, out_path, '')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('segmentation_path', type=str, help='path to segmentation (hdf5)')
    parser.add_argument('segmentation_key', type=str, help='path in h5-file to segmentation')
    parser.add_argument('output_path', type=str, help='path for output')
    args = parser.parse_args()
    seg_path = args.segmentation_path
    assert os.path.exists(seg_path), seg_path
    seg_key = args.segmentation_key
    out_path = args.output_path
    assert out_path.split('.')[-1] in ('.tif', '.tiff'), "Output files need to be tif"
    prepare_upload(seg_path, seg_key, out_path)
