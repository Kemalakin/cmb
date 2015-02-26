#!/bin/python

"""
ilc.py
jlazear
1/16/15

Internal Linear Combination method of foreground cleaning.

Provides tools for generating CMB maps with foregrounds cleaned using
the Internal Linear Combination (ILC) method.

Example:

See main() of this file.
"""
__version__ = 20150116
__releasestatus__ = 'beta'

import numpy as np
import healpy as hp


def compute_ilc_weights(maps, masks=None):
    """
    Computes a set of ILC weights of a given set of `maps`, for each region
    specified by `mask`.

    `maps` should be a k x Npix array. Each of the k rows should be a map at
    one of the k different frequencies. All maps must have the same length.

    `masks` may be None, a single mask map, or a list/array of masks.

    If `masks` is None, then all pixels in `maps` are used to construct the
    ILC weights.

    If `masks` is a single map, uses only pixels in each of the `maps` that
    have a value of 1 in the mask.

    If `masks` is a list of masks, constructs an ILC weighting for each mask
    and returns an array of ILC weights.
    """
    nd = np.ndim(masks)
    if nd == 0:
        return _compute_ilc_weights(maps)
    elif np.ndim(masks) == 1:
        mask = np.logical_not(np.outer(np.ones(len(maps)), masks))
        maps = np.ma.masked_array(maps, mask)
        return _compute_ilc_weights(maps)
    else:
        ws = []
        for i in range(len(masks)):
            mask = np.logical_not(np.outer(np.ones(len(maps)), masks[i]))
            ms = np.ma.masked_array(maps, mask)
            w = _compute_ilc_weights(ms)
            ws.append(w)
        ws = np.array(ws)
        return ws


def _compute_ilc_weights(maps):
    """
    Helper function for compute_ilc_weights().

    Actually performs the covariance matrix inversion and weight estimation
    for a single set of maps. If the maps are masked, uses a masked
    covariance function, otherwise uses a standard. Always returns a 1D
    standard ndarray of length k, where k = len(maps) = # frequencies.
    """
    if np.ma.isMaskedArray(maps):
        covfunc = lambda x: np.ma.compress_cols(np.ma.cov(x, ddof=0))
    else:
        covfunc = lambda x: np.cov(x, ddof=0)
    cov = covfunc(maps)
    icov = np.linalg.pinv(cov)  # Naive inversion, since cov ~ k x k is small
    sumicov = icov.sum(0)
    totalicov = sumicov.sum()
    w = sumicov/totalicov
    return w


def ilc_map_from_weights(maps, weights, regions_map=None, sigma=1.5*np.pi/180,
                         return_weights_map=False):
    """
    Construct an ILC map from a set of raw temperature maps, a set of weights
    per region, and a map of regions.

    `maps` must have shape (Nfreq, Npix). `weights` must have shape
    (Nregions, Nfreq). `regions_map` must have shape (Npix,) and each pixel
    should contain the integer identifier to which region the pixel is
    assigned.

    If `regions_map` is None, simply performs the linear combination of the
    maps according to the weights. #TODO NEEDS TO BE IMPLEMENTED

    `sigma` is the smoothing factor to reduce edge effects. It is applied to
    each region's weight map before multiplying into the raw maps. See
    Bennett 2003 or Hinshaw 2007 for details. `sigma` is the kernel radius
    (standard deviation) in radians.

    If `return_weights_map` is True, then also returns the summed weight map
    as a diagnostic tool.

    All of the maps must be in RING format!
    """
    Thats = np.dot(weights, maps)
    That = np.zeros(Thats.shape[1])
    weights_map = np.zeros(Thats.shape[1])
    for i in range(len(Thats)):
        m = np.zeros_like(That)
        m[regions_map == i] = 1
        if sigma is not None:
            # hp.smoothing does not preserve the mean, so add it back in
            mbar = m.mean()
            m = hp.smoothing(m, sigma=sigma, verbose=False) + mbar
        That += Thats[i]*m
        weights_map += m
    That = That/weights_map
    if return_weights_map:
        return That, weights_map
    else:
        return That


def main():
    # Load data
    from lib.wmap import load_wmap_maps, load_wmap_masks

    bands = ['K', 'Ka', 'Q', 'V', 'W']
    fnames = ['../data/wmap_band_smth_imap_r9_9yr_{0}_v5.fits'.format(band)
              for band in bands]
    maskdata = '../data/wmap_ilc_rgn_defn_9yr_v5.fits'

    print "Loading WMAP data..."
    maps = load_wmap_maps(fnames)
    masks, region = load_wmap_masks(maskdata)

    # Compute weights
    print "Computing ILC weights..."
    nregions = masks.shape[0]
    import time
    t0 = time.time()
    ws = compute_ilc_weights(maps, masks)
    tf = time.time()
    print "Computed ILC weights for {0} regions in {1} seconds.".format(
        nregions, tf - t0)

    # Print weights
    print "ILC weights"
    linestr = "{0:^10} {1:^10} {2:^10} {3:^10} {4:^10} {5:^10}"
    print linestr.format(*(['GROUP'] + bands))
    for i in range(len(ws)):
        linelist = [i] + [str(w).ljust(8, '0')[:8] for w in ws[i]]
        print linestr.format(*linelist)

    # Construct ILC map
    print "Constructing ILC map from weights..."
    T_hat = ilc_map_from_weights(maps, ws, region, sigma=None)

    # Display ILC map
    print "Plotting ILC map. Close figure to exit."
    hp.mollview(T_hat, title='ILC foreground-cleaned map using WMAP9 data',
                unit='mK', min=-.452123, max=.426628)
    from matplotlib import pylab
    pylab.show()


if __name__ == "__main__":
    main()
