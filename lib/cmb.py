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
import subprocess
import tempfile

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


def get_dls(lensed=True, fname=None):
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

    By default uses the file 'data/lcdm_planck_2013_cl_lensed.dat',
    which must be located in the data directory. If lensed is false,
    uses 'data/lcdm_planck_2013_cl.dat' If a `fname` is specified, searches for
    that file and attempts to use it. Note that all paths are specified
    relative to the base directory of the cmb package.
    """
    if fname is None:
        fname = 'data/lcdm_planck_2013_cl.dat'
    if fname.endswith('.dat'):
        fname, _ = fname.rsplit('.', 1)
    if (not fname.endswith('_lensed')) and lensed:
        fname = fname + '_lensed'
    clsfname = cmbpath + fname + '.dat'
    try:
        cls = np.loadtxt(clsfname)
    except IOError:
        fnamebase = fname.rsplit('.', 1)[0]
        fnamebase = fnamebase.rsplit('_lensed', 1)[0]
        fnamebase = fnamebase.rsplit('_cl', 1)[0]
        try:
            generate_cls(fnamebase + '.ini', output=fnamebase)
        except IOError:
            print generate_cls(fnamebase + '_parameters.ini',
                               output=fnamebase + '_')
        cls = np.loadtxt(clsfname)

    pdict = {}
    names = ['ell', 'TT', 'EE', 'BB', 'TE']#, 'dd', 'dT', 'dE']
    for i in range(len(names)):
        name = names[i]
        left = np.array([0, 1]) if name == 'ell' else np.array([0, 0])
        pdict[name]= np.hstack([left, cls[..., i]])

    return pdict


def get_cls(lensed=True, fname=None):
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

    By default uses the file 'data/lcdm_planck_2013_cl_lensed.dat',
    which must be located in the data directory. If lensed is false,
    uses 'data/lcdm_planck_2013_cl.dat' If a `fname` is specified, searches for
    that file and attempts to use it. Note that all paths are specified
    relative to the base directory of the cmb package.
    """
    pdict = get_dls(lensed=lensed, fname=fname)
    ells = pdict['ell'][1:]
    for key, value in pdict.items():
        if key != 'ell':
            pdict[key][1:] = 2*np.pi/(ells*(ells+1))*value[1:]
    return pdict


def generate_T_map(Nside=512, lensed=True, n2r=False, fname=None):
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
    Cl_TTs = get_cls(lensed=lensed, fname=fname)['TT']
    TTmap = hp.synfast(Cl_TTs, nside=Nside)
    if not n2r:
        TTmap = hp.reorder(TTmap, r2n=True)
    return TTmap


def generate_maps(Nside=512, lensed=True, n2r=False, return_cls=False,
                       fname=None, cal_gains=None):
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

    `cal_gains` determines if a calibration gain factor should be injected into
    the power spectrum before making the map. `cal_gains` should be a list with
    2 sub-lists. The first sublist contains the ells at which the gain should
    be applied, and the second sublist contains the gain factors corresponding
    to the ells. E.g.,

        cal_gains = [[10, 11, 12],        # Apply gain to ell = 10, 11, 12
                     [1.05, 1.1, 1.05]]   # Gains are 1.05, 1.1, 1.05

    Note, however, that if `return_cls` is True, the reported Cl's do NOT
    include the calibration gain.
    """
    cldict = get_cls(lensed=lensed, fname=fname)
    cls = np.array([cldict[xx] for xx in ('TT', 'EE', 'BB', 'TE')])
    if cal_gains is not None:
        ells = cldict['ell']
        gain_ells, gains = cal_gains
        for i in range(len(gain_ells)):
            gell = gain_ells[i]
            gain = gains[i]
            index = np.where(ells == gell)[0]
            cls[:, index] *= gain
    maps = list(hp.synfast(cls, nside=Nside, pol=True, new=True,
                           verbose=False))
    if not n2r:
        for i in range(len(maps)):
            maps[i] = hp.reorder(maps[i], r2n=True)
    if return_cls:
        return maps, cldict
    else:
        return maps


def generate_cls(input='data/lcdm_planck_2013_parameters.ini', output=None,
                 force_camb_format=True):
    """
    Uses CLASS to generate a set of Cl's corresponding to the cosmology
    specified by the `input` file.

    User must have `class` available on the path.

    All filenames are specified relative to the base directory of the cmb
    package.

    If `output` is specified, sets the root field of `class` to `output`
    before running `class`.
    """
    if (output is None) and (not force_camb_format):
        toexc = r'class {0}'.format(input)
        # Why shell=True required?
        return subprocess.check_output(toexc, cwd=cmbpath, shell=True)

    with tempfile.NamedTemporaryFile(suffix='.ini', bufsize=1) as tf:
        with open(input) as f:
            foundoutput = False
            foundformat = False
            for line in f:
                if (output is not None) and line.startswith('root'):
                    foundoutput = True
                    line = 'root = {0}\n'.format(output)
                elif force_camb_format and line.startswith('format'):
                    foundformat = True
                    line = 'format = camb\n'
                tf.write(line)

            if not foundoutput:
                tf.write('root = {0}\n'.format(output))
            if not foundformat:
                tf.write('format = camb\n')

        # Why shell=True required?
        return subprocess.check_output(r'class {0}'.format(tf.name),
                                       cwd=cmbpath, shell=True)