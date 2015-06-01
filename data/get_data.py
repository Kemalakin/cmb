#!/bin/python

"""
get_data.py
jlazear
1/6/15

Pull WMAP data from LAMBDA.

Example:

$ cd /path/to/cmb/data
$ python get_data.py

Make sure the cmb base directory is in your $PYTHONPATH.
"""
__version__ = 20150106
__releasestatus__ = 'beta'

import inspect
import os
import tarfile
import wget

import lib

# Path to the cmb/data/ directory.
datapath = os.path.dirname(os.path.abspath(inspect.getfile(lib))) + '/../data/'


fnames = [('http://lambda.gsfc.nasa.gov/data/map/dr5/skymaps/9yr/smoothed'
          '/wmap_band_smth_imap_r9_nineyear_v5.tar.gz'),
          ('http://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc/wmap_ilc_9yr_v5'
          '.fits'),
          ('http://lambda.gsfc.nasa.gov/data/map/dr5/dfp/ilc'
           '/wmap_ilc_rgn_defn_9yr_v5.fits'),
          ('http://irsa.ipac.caltech.edu/data/Planck/release_1/all-sky-maps'
           '/maps/COM_CompMap_dust-commrul_0256_R1.00.fits'),
          ('http://lambda.gsfc.nasa.gov/data/map/dr4/dfp/mcmc_maps'
           '/wmap_mcmc_k_synch_stk_q_7yr_v4p1.fits'),
          ('http://lambda.gsfc.nasa.gov/data/map/dr4/dfp/mcmc_maps'
           '/wmap_mcmc_k_synch_stk_u_7yr_v4p1.fits'),
          ('http://lambda.gsfc.nasa.gov/data/map/dr5/dcp/spectra'
           '/wmap_tt_spectrum_9yr_v5.txt'),
          ]

print "Downloading files..."
for fname in fnames:
    _, fn = os.path.split(fname)
    if not os.path.isfile(datapath + fn):
        print "Downloading file: {0}".format(fname)
        wget.download(fname, out=datapath+fn)

print "Extracting files..."
for fname in fnames:
    if fname.endswith('tar.gz'):
        _, fn = os.path.split(fname)
        print "Extracting file: {0}".format(fn)
        tar = tarfile.open(datapath + fn)
        tar.extractall(path=datapath)
        tar.close()