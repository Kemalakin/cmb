#!/bin/python

"""
noise.py
jlazear
3/11/15

Noise for PIPER.

Long description

Example:

<example code here>
"""
__version__ = 20150311
__releasestatus__ = 'beta'


import numpy as np
from scipy.integrate import quad


def dPdT(nu_min, nu_max, T=2.7260, eta_optical=0.42, eta_detector=0.5,
         A=0.058, Omega=1.02e-5, verbose=True):
    """
    Compute dP/dT for the specified frequency band. See docs/noise notes for
    detailed description.

    Returns a scalar value dP/dT, with units of W/K.

    Adjustable parameters and defaults (for PIPER):
        T=2.7260 - Sky temperature, in K. arxiv: 0911.1955v2 for default.
        eta_optical=0.42 - Optical efficiency from sky to detector. Unitless.
        eta_detector=0.5 - Detector absorption efficiency. Unitless.
        A=0.058 - Single detector area, in m^2. Etendue (A*Omega) is conserved,
                  so this value actually corresponds to area at dewar aperture.
                  This is allowed as long as Omega is measured at same spot.
                  Number from bjohnson memo.
        Omega=1.02e-5 - Single detector solid angle, in sr. Etendue (A*Omega) is
                  conserved, so this value actually corresponds to soli angle at
                  dewar aperture. This is allowed as long as A is measured at
                  same spot. Number from akogut dust code.

    The `verbose` flag prints out the values as they are computed.
    """
    h = 6.62606957e-34   # J*s
    kb = 1.3806488e-23   # J/K
    c = 3.e8             # m/s

    eta = eta_optical*eta_detector

    if verbose:
        sqdeg = (180./np.pi)**2
        print "detector solid angle = {0} sr = {1} sq. deg.".format(Omega,
                                                                    Omega*sqdeg)

    AOmega_s = A*Omega  # m^2*sr, single detector

    a1 = 2*h/c
    a2 = kb*T/h

    xmin = nu_min/a2
    xmax = nu_max/a2

    # coeff = AOmega_s*a1*(a2**4)*eta
    # Pintegrand = lambda x: x**3/(np.exp(x) - 1)
    # Pintegral = quad(Pintegrand, xmin, xmax)[0]
    #
    # P_T0 = coeff*Pintegral        # W

    coeff2 = AOmega_s*(2*kb/(c**2))*(kb*T/h)**3
    dPdTintegrand = lambda x: x**4*np.exp(x)/(np.exp(x) - 1)**2
    dPdTintegral = quad(dPdTintegrand, xmin, xmax)[0]

    dPdT_T0 = coeff2*dPdTintegral   # W/K

    if verbose:
        print "P @ T_0 = {0} pW".format(P_T0*1.e12)
        print "dP/dT @ T_0 = {0} (10^-18 W)/uK".format(dPdT_T0*1.e-6*1.e18)

    return dPdT_T0