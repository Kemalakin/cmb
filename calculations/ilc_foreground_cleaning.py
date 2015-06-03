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


import inspect
import os
import collections

import numpy as np
import healpy as hp

import lib


datapath = os.path.abspath(os.path.dirname(os.path.abspath(inspect.getfile(
    lib))) + '/../data/') + '/'


def regenerate_dust(frequencies, fgfile=None, Nside=512, verbose=True, force=False):
    """
    Perform only the dust map generation step of the ILC simulation.
    """
    if not force:
        try:
            np.load(fgfile)
            if verbose:
                print "Dust map {0} already exists.".format(fgfile)
            return
        except IOError:
            pass
    if verbose:
        print "Failed to load dust from file: {0}".format(fgfile)
        print "Generating new dust maps."
    # 3 deg smoothed QU polarized synchrotron-tracking dust maps
    MJypsr = lib.conversions.MJypsr  # W/(m^2 Hz sr) per MJy/sr
    convs = [1.e6/(lib.conversions.dBdT(f)/MJypsr)  # uK_CMB/(MJy/sr)
             for f in frequencies]

    dustmaps = [lib.foregrounds.generate_synchro_traced_dust_QU_map(nu=nu,
                        Nside=Nside,  n2r=True, sigma=3.*np.pi/180)
                for nu in frequencies]
    dustmaps = [dustmaps[i]*convs[i] for i in range(len(dustmaps))]   # uK_CMB
    dustmaps = np.array(dustmaps)
    f = fgfile if fgfile else datapath + 'dust.npy'
    np.save(f, dustmaps)
    if verbose:
        print "Saving dust maps to: {0}".format(f)


def polarized_ilc_reconstruction(frequencies, Nside=512, fname=None,
                                 lensed=False, fgfile=None, regnoise=None,
                                 verbose=True, _debug=False,
                                 modcov=False,
                                 regeneratedust=False,
                                 cal_gains=None):
    """
    Perform an end-to-end simulation of ILC reconstruction.

    1) Generate a fake polarized CMB sky (with optional power spectrum
    calibration gain).
    2) Load/generate dust maps at all of the specified frequencies.
    3) Combine the dust maps and (optionally) add map noise.
    4) Perform ILC on total Q and U maps separately.
    5) Calculate C_l's from reconstructed Q/U maps.
    6) Package data into dictionary and return

    `frequencies` - Frequencies at which to construct maps and perform ILC,
        in Hz
    `Nside` - Size of HealPIX maps.
    `fname` - By default uses the file 'data/lcdm_planck_2013_cl_lensed.dat',
        which must be located in the data directory. If lensed is false,
        uses 'data/lcdm_planck_2013_cl.dat' If a `fname` is specified, searches
        for that file and attempts to use it. Note that all paths are specified
        relative to the base directory of the cmb package.
    `lensed` - See `fname`.
    `fgfile` - File to load dust maps from. Default is dust/dust.npy.
    `regnoise` - Instrument/regularization noise, in uK/(healpix pixel).
    `verbose` - Be talkative.
    `_debug` - Also perform ILC in foreground-background space, to faciliate
        the comparison of results.
    `regeneratedust` - Whether or not dust maps should be re-made for each
        instance. Generally should not, since the dust maps are not stochastic.
    `cal_gains` - Determines if a calibration gain factor should be injected
        into the power spectrum before making the map. `cal_gains` should be a
        list with 2 sub-lists. The first sublist contains the ells at which the
        gain should be applied, and the second sublist contains the gain factors
        corresponding to the ells. E.g.,

        cal_gains = [[10, 11, 12],        # Apply gain to ell = 10, 11, 12
                     [1.05, 1.1, 1.05]]   # Gains are 1.05, 1.1, 1.05

    """
    frequencies = np.array(frequencies)
    if verbose:
        print '-'*80
        print "Performing ILC simulation with frequencies: {0} GHz".format(
            frequencies/1.e9)

    # Construct CMB maps from spectra
    if verbose:
        print "Constructing CMB temperature and polarization maps."
    (Tmap, Qmap, Umap), cldict = lib.cmb.generate_maps(Nside=Nside,
                        n2r=True, return_cls=True, fname=fname, lensed=lensed,
                        cal_gains=cal_gains)
    cmbQUmaps = np.vstack([Qmap, Umap])  # uK

    # Try to load dust from file.
    if fgfile is None:
        fgfile = datapath + 'dust.npy'
    try:
        dustmaps = np.load(fgfile)
        if regeneratedust or (len(dustmaps) != len(frequencies)):
            raise IOError
        if verbose: print "Loaded dust from file: {0}".format(fgfile)
    except (AttributeError, IOError):
        regenerate_dust(frequencies, fgfile, Nside, verbose, force=True)
        # if verbose:
        #     print "Failed to load dust from file: {0}".format(fgfile)
        #     print "Generating new dust maps."
        # # 3 deg smoothed QU polarized synchrotron-tracking dust maps
        # MJypsr = lib.conversions.MJypsr  # W/(m^2 Hz sr) per MJy/sr
        # convs = [1.e6/(lib.conversions.dBdT(f)/MJypsr)  # uK_CMB/(MJy/sr)
        #          for f in frequencies]
        #
        # dustmaps = [lib.foregrounds.generate_synchro_traced_dust_QU_map(nu=nu,
        #                     Nside=Nside,  n2r=True, sigma=3.*np.pi/180)
        #             for nu in frequencies]
        # dustmaps = [dustmaps[i]*convs[i] for i in range(len(dustmaps))]   # uK_CMB
        # dustmaps = np.array(dustmaps)
        # f = fgfile if fgfile else datapath + 'dust.npy'
        # np.save(f, dustmaps)
        # if verbose:
        #     print "Saving dust maps to: {0}".format(f)

    # Construct CMB + foreground maps
    if verbose:
        print "Combining maps."
    if regnoise is not None:
        if not isinstance(regnoise, collections.Iterable):
            regnoise = [regnoise]*len(dustmaps)
        if verbose:
            print "Instrument/regularization noise with std dev {0} uK".format(
                regnoise)
        # noise = regnoise*np.random.randn(*np.shape(cmbQUmaps[0]))
    else:
        regnoise = [0.]*len(dustmaps)
        if verbose:
            print "No instrument/regularization noise."

    totalQUmaps = []
    noisemaps = []
    for i in range(len(dustmaps)):
        dustmap = dustmaps[i]
        rn = regnoise[i]
        noise = rn*np.random.randn(*np.shape(cmbQUmaps))
        noisemaps.append(noise)
        totalQUmaps.append(cmbQUmaps + dustmap + noise)
    noisemaps = np.array(noisemaps)
    # totalQUmaps = [cmbQUmaps + dustmap + noise for dustmap in dustmaps]

    # Perform ILC on each set of Q and U maps
    #
    # # Get WMAP masks and regions
    # maskdata = 'data/wmap_ilc_rgn_defn_9yr_v5.fits'
    # masks, region = load_wmap_masks(maskdata)
    # ws = compute_ilc_weights(maps, masks)

    totalQmaps = np.vstack([totalQUmaps[i][0] for i in range(len(totalQUmaps))])
    totalUmaps = np.vstack([totalQUmaps[i][1] for i in range(len(totalQUmaps))])

    if verbose: print "Performing ILC on Q/U maps separately."
    if modcov:
        print ("Using modified covariance matrix (Efstathiou 2009): F -> F - "
               "diag(var(maps)).")
    Qweights = lib.ilc.compute_ilc_weights(totalQmaps, modcov=modcov)
    Uweights = lib.ilc.compute_ilc_weights(totalUmaps, modcov=modcov)

    if verbose: print "Reconstructing ILC Q/U maps."
    ilcQmap = np.dot(Qweights, totalQmaps)
    ilcUmap = np.dot(Uweights, totalUmaps)

    if verbose: print "Computing C_l's from reconstructed maps."
    recon_cls = hp.anafast([Tmap, ilcQmap, ilcUmap])  # TT, EE,  BB, TE, EB, TB
    labels = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
    recon_cldict = {labels[i]: recon_cls[i] for i in range(len(labels))}
    recon_cldict['ell'] = np.arange(len(recon_cldict['TT']), dtype='float')

    retdict = {'cl_in': cldict, 'cl_out': recon_cldict,
               'cmbTmap': Tmap, 'cmbQUmaps': cmbQUmaps, 'dustmaps': dustmaps,
               'noisemaps': noisemaps,
               'totalQUmaps': totalQUmaps,
               'weights_Q': Qweights, 'weights_U': Uweights,
               'ilcQmap': ilcQmap, 'ilcUmap': ilcUmap,
               'frequencies': frequencies, 'regnoise': regnoise}

    if _debug:
        if verbose:
            print ("Performing analytical ILC simulation in "
                  "foreground/background space")
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

    if verbose:
        print "Done with ILC with frequencies {0} GHz!".format(frequencies/1.e9)
        print '-'*80
    return retdict
