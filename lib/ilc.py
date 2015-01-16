#!/bin/python

"""
ilc.py
jlazear
1/6/15

One-line description

Long description

Example:

<example code here>
"""
__version__ = 20150106
__releasestatus__ = 'beta'

import numpy as np
import healpy as hp
from astropy.io import fits


def load_wmap_maps(fnames, n2r=True):
    """
    Load WMAP-format maps from file (i.e. from LAMBDA).

    Returns an n x Npix array, where n = len(`fnames`) is the number of maps
    and Npix is the number of pixels in each map. All maps read in at one
    time must have the same length (Npix).

    If `fnames` is a single file and not a list, reads and returns the single
    specified map.

    If `n2r` is True, converts from NEST format (the WMAP standard) to RING
    format (the healpy standard).
    """
    try:
        Nbands = np.shape(fnames)[0]
    except IndexError:
        # Single-map case. Just read and return the map.
        with fits.open(fnames) as f:
            a = f[1].data['TEMPERATURE']
        if n2r:
            a = hp.reorder(a, n2r=True)
        return a

    fname0 = fnames[0]

    # Initialize array and populate first band
    with fits.open(fname0) as f0:
        Npix = f0[1].header['NAXIS2']
        a = np.empty([Nbands, Npix])
        a[0] = f0[1].data['TEMPERATURE']

    for i, fname in enumerate(fnames[1:]):
        with fits.open(fname) as f:
            a[i+1] = f[1].data['TEMPERATURE']

    if n2r:
        for i in range(a.shape[0]):
            a[i] = hp.reorder(a[i], n2r=True)

    return a


def load_wmap_masks(fname='data/wmap_ilc_rgn_defn_9yr_v5.fits', n2r=True):
    """
    Loads the WMAP ILC region definition files and constructs the region masks.

    Returns `masks`, `region`.

    `masks` is a 12 x Npix uint16 array, each row of which is a mask map.
    For region `i`, all pixels in `masks`[i] that are 1 should contribute to
    the ILC weighting.

    `region` is a map describing to which region each pixel belongs. The ILC
    solution for region `i` (i.e. using pixels from `masks`[i]) should be
    assigned to pixels whose value is `i`.

    Note that region 0 uses pixels primarily from the galaxy but applies its
    solution to pixels primarily away from the galaxy. This is correct.
    """
    with fits.open(fname) as f:
        fdata = f[1]
        bitmask = fdata.data['TEMPERATURE'].astype('uint16')
        region = fdata.data['N_OBS'].astype('uint16')
        if n2r:
            bitmask = hp.reorder(bitmask, n2r=True)
            region = hp.reorder(region, n2r=True)

    masks = np.empty([12, len(bitmask)])
    for i in range(12):
        masks[i] = (bitmask >> i) & 1
    return masks, region


def compute_ilc_weights(maps, masks=None):
    """
    Computes a set of ILC weights of a given set of `maps`, for each region
    specified by `mask`.

    `maps` should be a k x Npix array. Each of the k rows should be a map at
    one of the k different frequencies. All maps must have the same length.

    `masks` may be None, a single mask map, or a list/array of masks. If
    `masks` is None, then all pixels in `maps` are used to construct the ILC
    weights. If `masks` is a single map, uses only pixels in each of the
    `maps` that have a value of 1 in the mask. If `masks` is a list of masks,
    constructs an ILC weighting for each mask and returns an array of ILC
    weights.
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
        covfunc = lambda x: np.ma.compress_cols(np.ma.cov(x))
    else:
        covfunc = np.cov
    cov = covfunc(maps)
    icov = np.linalg.pinv(cov)  # Naive inversion, since cov ~ k x k is small
    sumicov = icov.sum(0)
    totalicov = sumicov.sum()
    w = sumicov/totalicov
    return w


def ilc_map_from_weights(maps, weights, regions_map, sigma=1.5*np.pi/180,
                         return_weights_map=False):
    """
    Construct an ILC map from a set of raw temperature maps, a set of weights
    per region, and a map of regions.

    `maps` must have shape (Nfreq, Npix). `weights` must have shape
    (Nregions, Nfreq). `regions_map` must have shape (Npix,) and each pixel
    should contain the integer identifier to which region the pixel is
    assigned.

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