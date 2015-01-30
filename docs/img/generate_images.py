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
    fig.clf()


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
    fig.clf()


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
    fig.clf()

    fig = plt.figure(1)
    hp.mollview(Umap, fig=1, nest=True, norm='hist', unit='MJy/sr', title='')
    print strtemplate.format('Simple U map', fnameU)
    fig.savefig(fnameU)
    fig.clf()


#gammaQU
fnames = [imgpath + 'gamma.png', imgpath + 'gammaQ.png', imgpath + 'gammaU.png']
allthere = all(map(os.path.isfile, fnames))
if allthere and not force:
    for fname in fnames:
        print strexists.format(fname)
else:
    # dust353map = lib.foregrounds.generate_simple_dust_map()
    # pfrac = 0.2
    # poldustmap = pfrac*dust353map
    # mapnames = ['wmap_mcmc_k_synch_stk_q_7yr_v4p1.fits',
    #             'data/wmap_mcmc_k_synch_stk_u_7yr_v4p1.fits']
    # wmapQ, wmapU = lib.wmap.load_wmap_maps_QU(mapnames, Nside=256, n2r=True)
    # gamma = lib.foregrounds.generate_polarization_angle_map(wmapQ, wmapU,
    #                                                         sigma=3.*np.pi/180)
    # Qmap = poldustmap*np.cos(2*gamma)
    # Umap = poldustmap*np.sin(2*gamma)

    QUmaps = lib.foregrounds.generate_synchro_traced_dust_QU_map(
        sigma=3.*np.pi/180)
    gamma = 0.5*np.arctan2(-QUmaps[0], QUmaps[1])

    fnamegamma, fnameQ, fnameU = fnames
    fig = plt.figure(1)
    hp.mollview(gamma, fig=1, nest=True, unit='radians', title='')
    print strtemplate.format('Pol Dust Angle map', fnamegamma)
    fig.savefig(fnamegamma)
    fig.clf()

    fig = plt.figure(1)
    hp.mollview(QUmaps[0], fig=1, nest=True, unit='MJy/sr', norm='hist',
                title='')
    print strtemplate.format('Q map from synch angle', fnameQ)
    fig.savefig(fnameQ)
    fig.clf()

    fig = plt.figure(1)
    hp.mollview(QUmaps[1], fig=1, nest=True, unit='MJy/sr', norm='hist',
                title='')
    print strtemplate.format('U map from synch angle', fnameU)
    fig.savefig(fnameU)
    fig.clf()
