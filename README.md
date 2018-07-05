Mutex Watershed
===============

The Mutex Watershed algorithm for efficient segmentation without seeds.
See:
- TODO link to arxiv paper
- TODO link to blog


Installation
------------

**On Unix (Linux, OS X)**
 - create conda env with xtensor:
 - `conda create -n mws -c conda-forge xtensor-python`
 - activate the env:
 - `source activate mws`
 - clone this repository, enter it and install:
 - `python setup.py install`



ISBI Experiments
----------------

To reproduce the ISBI experiments, go to the `experiments/isbi` folder
and run the `isbi_experiments` script:
`python isbi_experiments.py /path/to/raw.h5 /path/to/affinities.h5 /path/to/res_folder --algorithms mws`

You can also reproduce the baseline results by specifying further algorithms.
Note that most of these will need the https://github.com/constantinpape/cremi_tools
repository and further dependencies specified there.

You can find the data used in the paper in `data/isbi`.


BSD Experiments
---------------

TODO
