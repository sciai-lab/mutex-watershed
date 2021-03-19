# ISBI Data

Contains the ISBI challenge data used for the Mutex Watershed paper.
The file `isbi_train_volume.h5` contains
- `/raw` - the training voulem raw data
- `/labels/membranes` - the original membrane labels-
- `/labels/gt_segmentation` - the segmentation groundtruth used for network training
- `/affinities` - the affinity predictions

The file `isbi_test_volume.h5` contains
- `/raw` - the test voulem raw data
- `/affinities` - the affinity predictions

The file `isbi_3d_model.pytorch` contains the weights of the
trained model (pytorch 0.2).

You can download the zip from
https://oc.embl.de/index.php/s/sXJzYVK0xEgowOz
