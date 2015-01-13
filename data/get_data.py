#!/bin/python

"""
get_data.py
jlazear
1/6/15

Pull WMAP data from LAMBDA.

WARNING: Do not use without first setting the current working directory to
the data directory! This script does not attempt to set the path correctly
and will happily overwrite files if you have the directory set incorrectly.

Example:

$ cd /path/to/cmb/data
$ python get_data.py
"""
__version__ = 20150106
__releasestatus__ = 'beta'

import os
import tarfile
import wget

fnames = [('http://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/smoothed'
          '/wmap_band_smth_imap_r9_nineyear_v5.tar.gz'),
          ('http://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc/wmap_ilc_9yr_v5'
          '.fits'),]

print "Downloading files..."
for fname in fnames:
    _, fn = os.path.split(fname)
    if not os.path.isfile(fn):
        print "Downloading file: {0}".format(fname)
        wget.download(fname)

print "Extracting files..."
for fname in fnames:
    if fname.endswith('tar.gz'):
        _, fn = os.path.split(fname)
        print "Extracting file: {0}".format(fn)
        tar = tarfile.open(fn)
        tar.extractall()
        tar.close()