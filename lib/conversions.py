#!/bin/python

"""
conversions.py
jlazear
1/20/15

Unit conversions for working with CMB data.

Example:

<example code here>
"""
__version__ = 20150120
__releasestatus__ = 'beta'


import numpy as np


c = 299792458.      # m/s
h = 6.62606957e-34  # m^2*kg/s
kB = 1.3806488e-23  # m^2*kg/s^2 K

MJypsr = 1.e-20     # (W/m^2 sr Hz) per MJy/sr


def B(T, nu):
    """
    The Planck distribution. Converts from thermodynamic temperature (K)
    units to spectral intensity (W/m^2 sr Hz) units.

    Divide the output of this function by MJypsr (=1.e-20) to get spectral
    intensity in MJy/sr.

    Returns the spectral intensity in W/m^2 sr Hz.
    """
    return (2*h*nu**3/(c**2)) / (np.exp(h*nu/(kB*T)) - 1)


def Binv(I, nu):
    """
    The inverse of the Planck distribution function. Converts from spectral
    intensity units (W/m^2 sr Hz) to thermodynamic temperature units (K).

    If the input is in MJy/sr, you must first multiply by MJypsr (=1.e-20)
    before applying this function.

    Returns the thermodynamic temperature in K.
    """
    return (h*nu/kB) / np.log(1 + 2*h*nu**3/(I*c*c))


def Binvapprox(I, nu):
    """
    The h*nu/kB*T << 1 limit of the inverse of the Planck distribution
    function. Converts from spectral intensity units (W/m^2 sr Hz) to antenna
    temperature units (K_antenna).

    If the input is in MJy/sr, you must first multiply by MJypsr (=1.e-20)
    before applying this function.

    Returns the antenna temperature in K_antenna.
    """
    return c*c*I/(2*kB*nu*nu)


def Bapprox(T_A, nu):
    """
    The h*nu/kB*T << 1 limit of the Planck distribution function. Converts
    from antenna temperature units (K_antenna) to spectral intensity units
    (W/m^2 sr Hz).

    Divide the output of this function by MJypsr (=1.e-20) to get spectral
    intensity in MJy/sr.

    Returns the spectral intensity in W/m^2 sr Hz.
    """
    return 2*nu*nu*kB*T_A/(c*c)

def dBdT(nu, T=2.72548):
    """
    The first order conversion factor between CMB temperature units (K_CMB) and
    spectral intensity units (W/m^2 sr Hz), dB/dT.

    The reference temperature may be set with `T`, and is by default the CMB
    temperature (Fixsen 2009).

    Divide the output of this function by MJypsr (=1.e-20) to get spectral
    intensity in terms of MJy/sr.

    Since this is a linear conversion (in intensity or temperature units), one
    may get the inverse conversion with 1/dBdT.

    Returns the spectral intensity per CMB temperature in (W/m^2 sr Hz)/K_CMB.
    """
    x = np.exp(h*nu/(kB*T))
    return 2*h*h*(nu**4)/(kB*c*c*T*T) * x/((x - 1)**2)