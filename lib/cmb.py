#!/bin/python

"""
cmb.py
jlazear
2/3/15

Generates C_l's.

NOTE: This module is currently minimally functional. The Python wrapper of
CLASS is significantly less powerful and easy to use than the C version.
Currently, this module only allows the user to import the Lambda-CDM model from
the CLASS default lcdm parameters. It should be expanded someday...
"""
__version__ = 20150203
__releasestatus__ = 'beta'

import os

import numpy as np
import healpy as hp

# Path to the cmb/data/ directory.
libpath = os.path.dirname(os.path.abspath(__file__)) + '/'
cmbpath = os.path.abspath(libpath + '/../') + '/'
datapath = os.path.abspath(cmbpath + '/data/') + '/'


# NOTE: many parameters that are valid in the C interface are invalid for the
#       Python interface. Without a good understanding of which are which, this
#       function is difficult to get working.
def get_params_from_file(fname):
    """
    docstring
    """
    paramdict = {}
    ignore_keys = ('root', 'write parameters', 'write primordial',
                   'write warnings', 'write background',
                   'write thermodynamics')
    with open(fname, 'r') as f:
        for line in f:
            line = line.strip()
            if (not line.startswith('#')) and '=' in line:
                sline = line.split('=', 1)
                key = sline[0].strip()
                try:
                    value = float(sline[1])
                except ValueError:
                    value = sline[1].strip()
                if key not in ignore_keys:
                    paramdict[key] = value
    return paramdict


def get_lcdm_dls(lensed=True):
    """
    Returns a dictionary of Dl's in units of uK^2.

    If `lensed` is True, returns lensed Dl's. Otherwise, returns unlensed Dl's.
    These Dl's are in units of uK^2 and are related to Cl's by

        D_\ell^{XX} = \frac{\ell(\ell+1)}{2\pi} C_\ell^{XX}

    The Dl's are determined by the default LCDM parameters of CLASS, which
    should agree with Planck 2013. The specific parameter values (as well as
    the input to CLASS) may be found in data/lcdm_parameters.ini.

    Returns a dictionary with keys ('ell', 'TT', 'EE', 'TE', 'BB'). \ell ranges
    from 0 to 3500 if `lensed` is True, and 0 to 3500 if `lensed` is False.
    The format of the Dl's should match that of CAMB.

    Note that the data is extended to include the ell = 0 and ell = 1 bins,
    since healpy.synfast expects the input Cl's to range from 0 to lmax. The
    extended data values are simply 0.

    Also note that these must be converted to Cl's (see above formula) before
    being passed into healpy.synfast.
    """
    lstr = '_lensed' if lensed else ''
    clsfname = datapath + 'lcdm_planck_2013_cl{0}.dat'.format(lstr)
    # lcdmparams = datapath + 'lcdm_planck_2013_parameters.ini'
    cls = np.loadtxt(clsfname)
    # NOTE: The next 3 lines are only necessary if the Cl data files are in
    #       CLASS format. They are currently in CAMB format, so unnecessary.
    # pdict = get_params_from_file(lcdmparams)
    # T_cmb = pdict['T_cmb']*1.e6   # uK
    # cls[..., 1:5] = cls[..., 1:5]*T_cmb*T_cmb   # uK^2

    pdict = {}
    names = ['ell', 'TT', 'EE', 'BB', 'TE']#, 'dd', 'dT', 'dE']
    for i in range(len(names)):
        name = names[i]
        left = np.array([0, 1]) if name == 'ell' else np.array([0, 0])
        pdict[name]= np.hstack([left, cls[..., i]])

    return pdict


def get_lcdm_cls(lensed=True):
    """
    Returns a dictionary of Cl's in units of uK^2.

    If `lensed` is True, returns lensed Cl's. Otherwise, returns unlensed Cl's.
    These Cl's are in units of uK^2 and are related to Dl's by

        D_\ell^{XX} = \frac{\ell(\ell+1)}{2\pi} C_\ell^{XX}

    The Dl's are determined by the default LCDM parameters of CLASS, which
    should agree with Planck 2013. The specific parameter values (as well as
    the input to CLASS) may be found in data/lcdm_parameters.ini.

    Returns a dictionary with keys ('ell', 'TT', 'EE', 'TE', 'BB'). \ell ranges
    from 0 to 3000 if `lensed` is True, and 0 to 3500 if `lensed` is False. The
    format of the Cl's should match that of CAMB.

    Note that the data is extended to include the ell = 0 and ell = 1 bins,
    since healpy.synfast expects the input Cl's to range from 0 to lmax. The
    extended data values are simply 0.
    """
    pdict = get_lcdm_dls(lensed=lensed)
    ells = pdict['ell'][1:]
    for key, value in pdict.items():
        if key != 'ell':
            pdict[key][1:] = 2*np.pi/(ells*(ells+1))*value[1:]
    return pdict


def generate_lcdm_T_map(Nside=512, lensed=True, n2r=False):
    """
    Generates a realization of the standard LCDM CMB temperature map,
    using parameters that should agree with Planck 2013, in units of uK,
    in thermodynamic units.

    The map is in Healpix format. If `n2r` is True, the resulting map will be
    in RING format. Otherwise, the resulting map will be in NEST format.

    `Nside` determines the number of pixels in the map. Npix = 12*Nside^2.

    If `lensed` is True, uses the lensed Cl's. Otherwise uses the unlensed
    Cl's.
    """
    Cl_TTs = get_lcdm_cls(lensed=lensed)['TT']
    TTmap = hp.synfast(Cl_TTs, nside=Nside)
    if not n2r:
        TTmap = hp.reorder(TTmap, r2n=True)
    return TTmap


def generate_lcdm_maps(Nside=512, lensed=True, n2r=False, return_cls=False):
    """
    Generates a realization of the standard LCDM CMB T, Q, and U maps,
    using parameters that should agree with Planck 2013, in units of uK,
    in thermodynamic units.

    The maps are in Healpix format. If `n2r` is True, the resulting maps will be
    in RING format. Otherwise, the resulting maps will be in NEST format.

    `Nside` determines the number of pixels in each map. Npix = 12*Nside^2.

    If `lensed` is True, uses the lensed Cl's. Otherwise uses the unlensed
    Cl's.

    If `return_cls` is True, the dictionary of Cl's is also returned. These
    Cl's are in units of uK^2 and are related to Dl's by

        D_\ell^{XX} = \frac{\ell(\ell+1)}{2\pi} C_\ell^{XX}
    """
    cldict = get_lcdm_cls(lensed=lensed)
    cls = [cldict[xx] for xx in ('TT', 'EE', 'BB', 'TE')]
    maps = list(hp.synfast(cls, nside=Nside, pol=True, new=True))
    if not n2r:
        for i in range(len(maps)):
            maps[i] = hp.reorder(maps[i], r2n=True)
    if return_cls:
        return maps, cldict
    else:
        return maps

