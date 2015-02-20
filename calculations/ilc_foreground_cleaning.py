#!/bin/python

"""
ilc_foreground_cleaning.py
jlazear
2/11/15

One-line description

Long description

Example:

<example code here>
"""
__version__ = 20150211


import numpy as np
import healpy as hp

import lib


def polarized_ilc_reconstruction(frequencies, Nside=512):
    frequencies = np.array(frequencies)

    # Construct CMB maps from spectra
    (Tmap, Qmap, Umap), cldict = lib.cmb.generate_lcdm_maps(Nside=Nside,
                        n2r=True, return_cls=True)
    cmbQUmaps = np.vstack([Qmap, Umap])  # uK

    # 3 deg smoothed QU polarized synchrotron-tracking dust maps
    MJypsr = lib.conversions.MJypsr
    convs = [1.e6/(lib.conversions.dBdT(f)/MJypsr)  # uK_CMB/(MJy/sr)
             for f in frequencies]

    dustmaps = [lib.foregrounds.generate_synchro_traced_dust_QU_map(nu=nu,
                        Nside=Nside,  n2r=True, sigma=3.*np.pi/180)
                for nu in frequencies]
    dustmaps = [dustmaps[i]*convs[i] for i in range(len(dustmaps))]   # uK_CMB

    # Construct CMB + foreground maps
    totalQUmaps = [cmbQUmaps + dustmap for dustmap in dustmaps]

    # Perform ILC on each set of Q and U maps
    #
    # # Get WMAP masks and regions
    # maskdata = 'data/wmap_ilc_rgn_defn_9yr_v5.fits'
    # masks, region = load_wmap_masks(maskdata)
    # ws = compute_ilc_weights(maps, masks)

    totalQmaps = np.vstack([totalQUmaps[i][0] for i in range(len(totalQUmaps))])
    totalUmaps = np.vstack([totalQUmaps[i][1] for i in range(len(totalQUmaps))])

    Qweights = lib.ilc.compute_ilc_weights(totalQmaps)
    Uweights = lib.ilc.compute_ilc_weights(totalUmaps)

    ilcQmap = np.dot(Qweights, totalQmaps)
    ilcUmap = np.dot(Uweights, totalUmaps)

    recon_cls = hp.anafast([Tmap, ilcQmap, ilcUmap])  # TT, EE,  BB, TE, EB, TB
    labels = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
    recon_cldict = {labels[i]: recon_cls[i] for i in range(len(labels))}
    recon_cldict['ell'] = np.arange(len(recon_cldict['TT']), dtype='float')
    
    retdict = {'cl_in': cldict, 'cl_out': recon_cldict,
               'cmbTmap': Tmap, 'cmbQUmaps': cmbQUmaps, 'dustmaps': dustmaps,
               'totalQUmaps': totalQUmaps,
               'weights_Q': Qweights, 'weights_U': Uweights,
               'ilcQmap': ilcQmap, 'ilcUmap': ilcUmap}

    return retdict