#!/bin/python

"""
generate_images.py
jlazear
1/22/15

Generates images used in the docs files.

Example:

$ python generate_images.py
"""
__version__ = 20150122
__releasestatus__ = 'beta'


import os
import sys
import argparse

import wget
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

imgpath = os.path.dirname(os.path.abspath(__file__)) + '/'
cmbpath = os.path.abspath(imgpath + '/../../') + '/'
sys.path.append(cmbpath)
import lib


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--regenerate",
                    help="force the regeneration of all figures",
                    action="store_true")
args = parser.parse_args()
force = bool(args.regenerate)

strtemplate = "Saving {0} to {1}\r\n"
strexists = "File {0} already exists."

# -------------------------------------
# ---------- Generate images ----------
# -------------------------------------

# planck_dust_353GHz.png
fname = imgpath + 'planck_dust_353GHz.png'
if os.path.isfile(fname) and not force:
    print strexists.format(fname)
else:
    dust353map = lib.foregrounds.generate_simple_dust_map()
    fig = plt.figure(1)
    hp.mollview(dust353map, fig=1, nest=True, norm='hist', unit='MJy/sr',
                title='')

    print strtemplate.format('Planck 353 GHz dust map', fname)
    fig.savefig(fname)


# cb9_f22_M.png
cb9_f22_fname = imgpath + 'cb9_f22_M.png'
if os.path.isfile(cb9_f22_fname) and not force:
    print strexists.format(cb9_f22_fname)
else:
    try:
        os.remove(cb9_f22_fname)
    except OSError:
        pass
    cb9_f22_url = ('http://lambda.gsfc.nasa.gov/product/map/current/pub_papers'
                   '/nineyear/basic_results/images/med/cb9_f22_M.png')
    print strtemplate.format('CB9 Fig22', cb9_f22_fname)
    wget.download(cb9_f22_url, out=cb9_f22_fname)
    print


# pol_dust.png
fname = imgpath + 'pol_dust_353GHz.png'
if os.path.isfile(fname) and not force:
    print strexists.format(fname)
else:
    dust353map = lib.foregrounds.generate_simple_dust_map()
    pfrac = 0.2
    poldustmap = pfrac*dust353map
    fig = plt.figure(1)
    hp.mollview(poldustmap, fig=1, nest=True, norm='hist', unit='MJy/sr', title='')

    print strtemplate.format('Simple polarized dust map', fname)
    fig.savefig(fname)


# QU.png
fnameQ = imgpath + 'Q.png'
fnameU = imgpath + 'U.png'
if os.path.isfile(fnameQ) and os.path.isfile(fnameU) and not force:
    print strexists.format(fnameQ)
    print strexists.format(fnameU)
else:
    dust353map = lib.foregrounds.generate_simple_dust_map()
    pfrac = 0.2
    poldustmap = pfrac*dust353map
    thetamap = np.random.rand(len(dust353map))*2*np.pi  # [0, 1)*2pi = [0, 2pi)
    Qfact = np.cos(thetamap)
    Ufact = np.sin(thetamap)
    Qmap = poldustmap*Qfact
    Umap = poldustmap*Ufact

    fig = plt.figure(1)
    hp.mollview(Qmap, fig=1, nest=True, norm='hist', unit='MJy/sr', title='')
    print strtemplate.format('Simple Q map', fnameQ)
    fig.savefig(fnameQ)

    fig = plt.figure(1)
    hp.mollview(Umap, fig=1, nest=True, norm='hist', unit='MJy/sr', title='')
    print strtemplate.format('Simple U map', fnameU)
    fig.savefig(fnameU)

