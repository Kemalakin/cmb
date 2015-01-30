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

import numpy as np
from astropy.io import fits
import healpy as hp

import lib

# Path to the cmb/data/ directory. We'll need to read the Planck 353 GHz map.
datapath = os.path.dirname(os.path.abspath(inspect.getfile(lib))) + '/../data/'
planck353dustfname = datapath + 'COM_CompMap_dust-commrul_0256_R1.00.fits'
wmap23synchQfname = datapath + 'wmap_mcmc_k_synch_stk_q_7yr_v4p1.fits'
wmap23synchUfname = datapath + 'wmap_mcmc_k_synch_stk_u_7yr_v4p1.fits'


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


def generate_simple_dust_QU_map(nu=353.e9, polfrac=0.2, beta=1.6, Nside=256,
                                clean_map=True, n2r=False):
    """
    Generates simple Q and U dust maps at the specified frequency.

    Each pixel is treated independently. An angle theta is randomly sampled
    from a uniform distribution between 0 and 2pi. Then the Q and U
    components are generated from the random angle,

        theta sampled from Uniform(0, 2pi)
        Q = p*I(nu, beta)*cos(theta)
        U = p*I(nu, beta)*sin(theta)

    where p is the polarization fraction `polfrac`, `nu` is the frequency of
    the maps, and `beta` is the power law index used to construct the thermal
    dust emission map.

    Uses generate_simple_dust_map() to generate the map. See its docstring
    for information about how the simple dust maps are constructed.

    The map is constructed base off of an Nside=256 map, but this function
    will scale it to the desired Nside.

    If `clean_map` is True, replaces pixels with negative values with 0.

    If `n2r` is True, converts from the NESTED format to the RING format.

    Returns a 2 x N_pix ndarray. The first row (length 2 axis) is the Q map
    and the second row is the U map. The maps are in units of spectral
    intensity with units MJy/sr.
    """
    imap = generate_simple_dust_map(nu=nu, beta=beta, Nside=Nside,
                                    clean_map=clean_map, n2r=n2r)
    QUmaps = np.empty([2, imap.shape[0]])
    theta = np.random.rand(imap.shape[0])*2*np.pi
    QUmaps[0] = polfrac*imap*np.cos(theta)
    QUmaps[1] = polfrac*imap*np.sin(theta)

    return QUmaps


def generate_polarization_angle_map(Qmap, Umap, sigma=None):
    """
    Generates a polarization angle map from the specified Q and U maps.

    The Q and U maps must be in RING format if `sigma` is not None. The Q and U
    maps must have the same Nside.

    If `sigma` is not None, then the maps are smoothed by a Gaussian with width
    `sigma` (radians) before computing the angle.

    Returns a single map of the same Nside as the input Q and U maps with the
    polarization angle in radians, with -pi <= gamma < pi.

    The angle follows the convention of Delabrouille et al. 2012,

        gamma = (1/2)*arctan(-U, Q)
    """
    if sigma is not None:
        Qmap = hp.smoothing(Qmap, sigma=sigma, verbose=False)
        Umap = hp.smoothing(Umap, sigma=sigma, verbose=False)

    gamma = 0.5*np.arctan2(-Umap, Qmap)
    return gamma


def generate_synchro_traced_dust_QU_map(nu=353.e9, polfrac=0.2, beta=1.6,
                                        Nside=256, clean_map=True, n2r=False,
                                        sigma=None):
    """
    Generates a polarized dust intensity map at the specified frequency using
    WMAP synchrotron data to determine the polarization angle.

    The Q and U components are generated from the polarization angle according
    to

        Q = p*I(nu, beta)*cos(2*gamma)
        U = p*I(nu, beta)*sin(2*gamma)

    where p is the polarization fraction `polfrac`, `nu` is the frequency of
    the maps, and `beta` is the power law index used to construct the thermal
    dust emission map, and gamma is the polarization angle.

    The polarization angle is determined from the WMAP 23 GHz synchrotron Q and
    U maps according to

        gamma = 0.5*arctan2(-U_23, Q_23)

    following the convention of Delabrouille et al. 2012.

    If `sigma` is not None, then the maps are smoothed by a Gaussian with width
    `sigma` (radians) before computing the angle.

    If `clean_map` is True, replaces pixels with negative values with 0.

    If `n2r` is True, converts from the NESTED format to the RING format.

    Returns a 2 x N_pix ndarray. The first row (length 2 axis) is the Q map
    and the second row is the U map. The maps are in units of spectral
    intensity with units MJy/sr.
    """
    dust353map = generate_simple_dust_map(nu=nu, beta=beta, Nside=Nside,
                                          clean_map=clean_map, n2r=n2r)
    mapnames = [wmap23synchQfname, wmap23synchUfname]
    wmapQ, wmapU = lib.wmap.load_wmap_maps_QU(mapnames, Nside=Nside, n2r=True)
    gamma = lib.foregrounds.generate_polarization_angle_map(wmapQ, wmapU,
                                                            sigma=sigma)
    if not n2r:
        gamma = hp.reorder(gamma, r2n=True)

    QUmaps = np.empty([2, gamma.shape[0]])
    QUmaps[0] = polfrac*dust353map*np.cos(2*gamma)
    QUmaps[1] = polfrac*dust353map*np.sin(2*gamma)

    return QUmaps

