#!/bin/python

"""
wmap.py
jlazear
1/16/15

Tools for loading WMAP data from LAMBDA data files.

Example:

<example code here>
"""
__version__ = 20150116
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


def main():
    bands = ['K', 'Ka', 'Q', 'V', 'W']
    fnames = ['../data/wmap_band_smth_imap_r9_9yr_{0}_v5.fits'.format(band)
              for band in bands]
    maskdata = '../data/wmap_ilc_rgn_defn_9yr_v5.fits'

    print "Loading WMAP data..."
    maps = load_wmap_maps(fnames)
    masks, region = load_wmap_masks(maskdata)
    return maps, masks, region

if __name__ == "__main__":
    main()