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


def polarized_ilc_reconstruction(frequencies, Nside=512, fname=None,
                                 lensed=False, _debug=False):
    frequencies = np.array(frequencies)

    # Construct CMB maps from spectra
    (Tmap, Qmap, Umap), cldict = lib.cmb.generate_maps(Nside=Nside,
                        n2r=True, return_cls=True, fname=fname, lensed=lensed)
    cmbQUmaps = np.vstack([Qmap, Umap])  # uK

    # 3 deg smoothed QU polarized synchrotron-tracking dust maps
    MJypsr = lib.conversions.MJypsr  # W/(m^2 Hz sr) per MJy/sr
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
               'ilcQmap': ilcQmap, 'ilcUmap': ilcUmap,
               'frequencies': frequencies}

    if _debug:
        var_Q = np.var(Qmap)
        var_U = np.var(Umap)

        def covcoeff(x, y):
            return np.mean(x*y) - np.mean(x)*np.mean(y)

        def magnitude(A):
            wons = np.ones(len(A))
            return np.einsum("i,ij,j", wons, A, wons)

        X_Q = [covcoeff(Qmap, dustmaps[i][0]) for i in range(len(dustmaps))]
        X_Q = np.array([X_Q]).T  # Make sure shape is  N_freqs x 1
        X_U = [covcoeff(Umap, dustmaps[i][1]) for i in range(len(dustmaps))]
        X_U = np.array([X_U]).T  # Make sure shape is N_freqs x 1

        F_Q = np.cov(np.array(dustmaps)[..., 0])  # cov(R, R)
        F_U = np.cov(np.array(dustmaps)[..., 1])  # cov(R, R)

        ones = np.ones([len(dustmaps), 1])
        G_Q = F_Q + ones*X_Q.T
        G_Qinv = np.linalg.pinv(G_Q)
        G_Qcond = np.linalg.cond(G_Q)
        G_U = F_U + ones*X_U.T
        G_Uinv = np.linalg.pinv(G_U)
        G_Ucond = np.linalg.cond(G_U)

        Ginvone_Q = np.dot(G_Qinv, ones)
        Ginvone_U = np.dot(G_Uinv, ones)
        GinvX_Q = np.dot(G_Qinv, X_Q)
        GinvX_U = np.dot(G_Uinv, X_U)

        weights_Q = ((1. + np.dot(ones.T, GinvX_Q))/(np.dot(ones.T, Ginvone_Q))
                     * Ginvone_Q - GinvX_Q)
        weights_U = ((1. + np.dot(ones.T, GinvX_U))/(np.dot(ones.T, Ginvone_U))
                     * Ginvone_U - GinvX_U)

        ilcQmap2 = np.dot(weights_Q.flatten(), totalQmaps)
        ilcUmap2 = np.dot(weights_U.flatten(), totalUmaps)


        debugdict = {'X_Q': X_Q, 'X_U': X_U, 'F_Q': F_Q, 'F_U': F_U,
                     'G_Q': G_Q, 'G_U': G_U,
                     'G_Qinv': G_Qinv, 'G_Uinv': G_Uinv,
                     'G_Qcond': G_Qcond, 'G_Ucond': G_Ucond,
                     'weights_Q': weights_Q, 'weights_U': weights_U,
                     'var_Q': var_Q, 'var_U': var_U,
                     'ilcQmap': ilcQmap2, 'ilcUmap': ilcUmap2}
        retdict['debug'] = debugdict

    return retdict