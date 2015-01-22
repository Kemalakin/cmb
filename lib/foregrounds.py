#!/bin/python

"""
foregrounds.py
jlazear
1/20/15

Tools for constructing CMB foregrounds.

Long description

Example:

<example code here>
"""
__version__ = 20150120
__releasestatus__ = 'beta'


import inspect
import os

from astropy.io import fits
import healpy as hp

import lib

# Path to the cmb/data/ directory. We'll need to read the Planck 353 GHz map.
datapath = os.path.dirname(os.path.abspath(inspect.getfile(lib))) + '/../data/'
planck353dustfname = datapath + 'COM_CompMap_dust-commrul_0256_R1.00.fits'


def generate_simple_dust_map(nu=353.e9, beta=1.6, Nside=256, clean_map=True,
                             n2r=False):
    """
    Generates a dust thermal emission intensity map at the specified frequency
    `nu`.

    The generated map is a simple power law rescaling of the Planck 353 GHz
    Commander-Ruler dust component map using a power law scaling factor with
    index `beta` (=1.6 by default). Requires the Planck 353 GHz dust component
    map to be available in the data directory,
        cmb/data/COM_CompMap_dust-commrul_0256_R1.00.fits

    The intensity is given by

        I(p) = I_0(p)*(nu/nu0)^beta

    where I_0(p) is the base thermal emission intensity map (Planck 353 GHz),
    nu is the target frequency, nu0 = 353 GHz is the base map frequency,
    and beta is the spectral index. By default, the Planck map (and thus this
    map) is in the NESTED format.

    The map is constructed base off of an Nside=256 map, but this function
    will scale it to the desired Nside.

    If `clean_map` is True, replaces pixels with negative values with 0.

    If `n2r` is True, converts from the NESTED format to the RING format.

    Returns a map of spectral intensity I_\nu in MJy/sr.
    """
    with fits.open(planck353dustfname) as f:
        map = f[1].data['I']

    nu0 = 353.e9  # 353 GHz
    factor = (nu/nu0)**beta
    map = factor*map

    if Nside != 256:
        map = hp.ud_grade(map, Nside, order_in='NESTED', order_out='NESTED')

    if n2r:
        map = hp.reorder(map, n2r=True)

    if clean_map:
        map = clean_simple_dust_map(map)
    return map


def clean_simple_dust_map(map):
    """Replaces non-positive pixels with 0."""
    map[map <= 0] = 0.
    return map


# def generate_simple_dust_QU_map(nu=353.e9, beta=1.6, Nside=256, clean_map=True,
#                                 n2r=False):
#     """
#     Generates simple Q and U maps
#     """