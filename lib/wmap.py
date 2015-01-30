#!/bin/python

"""
wmap.py
jlazear
1/16/15

Tools for loading WMAP data from LAMBDA data files.

Example:

> map_K = load_wmap_maps('../data/wmap_band_smth_imap_r9_9yr_K_v5.fits')
"""
__version__ = 20150116
__releasestatus__ = 'beta'

import numpy as np
import healpy as hp
from astropy.io import fits


def load_wmap_maps(fnames, n2r=True):
    """
    Load WMAP-format temperature maps from file (i.e. from LAMBDA).

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


def load_wmap_masks(fname, n2r=True):
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

    masks = np.empty([12, len(bitmask)], dtype='bool')
    for i in range(12):
        masks[i] = (bitmask >> i) & 1
    return masks, region


def load_wmap_maps_QU(fnames, n2r=True, Nside=None):
    """
    Load WMAP-format polarization maps from file (i.e. from LAMBDA).

    Returns an n x Npix array, where n = len(`fnames`) is the number of maps
    and Npix is the number of pixels in each map. All maps read in at one
    time must have the same length (Npix).

    If `fnames` is a single file and not a list, reads and returns the single
    specified map.

    If `n2r` is True, converts from NEST format (the WMAP standard) to RING
    format (the healpy standard).

    Scales the map to the specified `Nside`. If None, does not rescale the map.
    All input maps must have the same base Nside.
    """
    colname = 'BESTFIT'
    try:
        Nbands = np.shape(fnames)[0]
    except IndexError:
        # Single-map case. Just read and return the map.
        with fits.open(fnames) as f:
            a = f[1].data[colname]
            base_Nside = f[1].header['NSIDE']
        if (Nside is not None) and (Nside != base_Nside):
            a = hp.ud_grade(a, Nside, order_in='NESTED', order_out='NESTED')
        if n2r:
            a = hp.reorder(a, n2r=True)
        return a

    fname0 = fnames[0]

    # Initialize array and populate first band
    with fits.open(fname0) as f0:
        Npix = f0[1].header['NAXIS2']
        base_Nside = f0[1].header['NSIDE']
        a = np.empty([Nbands, Npix])
        a[0] = f0[1].data[colname]

    for i, fname in enumerate(fnames[1:]):
        with fits.open(fname) as f:
            a[i+1] = f[1].data[colname]

    if (Nside is not None) and (Nside != base_Nside):
        b = np.empty([a.shape[0], hp.nside2npix(Nside)])
        for i in range(a.shape[0]):
            b[i] = hp.ud_grade(a[i], Nside, order_in='NESTED',
                               order_out='NESTED')
        a = b
    if n2r:
        for i in range(a.shape[0]):
            a[i] = hp.reorder(a[i], n2r=True)



    return a


# From Table 12 of Bennett et al 2013, ADS: 2013ApJS..208...20B
ilc_weights = np.array([[.1555, -.7572, -.2689, 2.2845, -.4138],
                        [.0375, -.5137, .0223, 2.0378, -.5839],
                        [.0325, -.3585, -.3103, 1.8521, -.2157],
                        [-.0910, .1741, -.6267, 1.5870, -.0433],
                        [-.0762, .0907, -.4273, .9707, .4421],
                        [.1998, -.7758, -.4295, 2.4684, -.4629],
                        [-.0880, .1712, -.5306, 1.0097, 0.4378],
                        [.1578, -.8074, -.0923, 2.1966, -.4547],
                        [.1992, -.1736, -1.8081, 3.7271, -.9446],
                        [-.0813, -.1579, -.0551, 1.2108, .0836],
                        [.1717, -.8713, -.1700, 2.8314, -.9618],
                        [.2353, -.8325, -.6333, 2.8603, -.6298]])

def main():
    bands = ['K', 'Ka', 'Q', 'V', 'W']
    fnames = ['../data/wmap_band_smth_imap_r9_9yr_{0}_v5.fits'.format(band)
              for band in bands]
    maskdata = '../data/wmap_ilc_rgn_defn_9yr_v5.fits'

    print "Loading WMAP data..."
    maps = load_wmap_maps(fnames)
    masks, region = load_wmap_masks(maskdata)

    hp.mollview(region, title='WMAP ILC Regions')
    summap = sum(masks[i]*(i+1) for i in range(len(masks)))
    hp.mollview(summap, title='WMAP ILC "bitmasks", i.e. sum(masks[i]*(i+1)')
    from matplotlib import pyplot
    pyplot.show()

    return maps, masks, region

if __name__ == "__main__":
    main()