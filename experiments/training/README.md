# Training ISBI affinities

Use `train.py` to train affinities on the [isbi challenge data](https://hcicloud.iwr.uni-heidelberg.de/index.php/s/6LuE7nxBN3EFRtL) or similar data.


## Set-up

Set-up a conda-environemtn with the necessary dependencies via
```
$ conda env create -f environment.yaml
```
Note that you might need to adjust the cuda version.

In addition, you will need to add [neurofire]() to the environment and 
install the mutex watershed repo.

You will also need to adapt the input paths in the config files, e.g. [here](https://github.com/hci-unihd/mutex-watershed/blob/master/experiments/training/template_config/data_config.yml#L8).
You can change the input window size [here](https://github.com/hci-unihd/mutex-watershed/blob/master/experiments/training/template_config/data_config.yml#L2)
and the offset pattern that is used [here](https://github.com/hci-unihd/mutex-watershed/blob/master/experiments/training/train.py#L171).
