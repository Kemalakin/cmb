#!/bin/python

"""
util.py
jlazear
6/8/15

Utilities for working with the calculation modules.

Example:

<example code here>
"""
__version__ = 20150608
__releasestatus__ = 'beta'

import inspect
import os

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

import lib
import calculations.ilc_foreground_cleaning
from calculations.calibration_gain import read_dict_from_hd5, save_dict_to_hd5

datapath = os.path.abspath(os.path.dirname(os.path.abspath(inspect.getfile(
    lib))) + '/../data/') + '/'

Omega_single = 7.45446e-6  # sr, from bjohnson report
Omega_pixel = hp.nside2pixarea(512)  # pixel solid angle in sr
Omega_array = 9.54171e-3  # sr, from bjohnson report
regnoise = 3./1.3*np.array([1.3, 1.9, 6.7, 110.])*np.sqrt(Omega_single/Omega_pixel)  # uK/pixel
fname = datapath + 'explanatory_cl.dat'
lensed = False
modcov = False

print("="*80)
print("Importing calibration_gain.")
cntpath = os.path.abspath(datapath + 'cldict_normal_test.hd5')
if os.path.exists(cntpath):
    print("cldict_normal_test already exists at {0}. Loading it.".format(cntpath))
    cldict_normal_test = read_dict_from_hd5(cntpath)
else:
    print("cldict_normal_test does not exist. Creating it.")
    cldict_normal_test = calculations.ilc_foreground_cleaning.polarized_ilc_reconstruction(
                        frequencies=np.array([200., 270., 350., 600.])*1.e9, fname=fname,
                        regnoise=regnoise, lensed=lensed, modcov=modcov, verbose=True)
    save_dict_to_hd5(cntpath, cldict_normal_test)
clnpath = os.path.abspath(datapath + 'cldict_noise.hd5')
if os.path.exists(clnpath):
    print("cldict_noise already exists at {0}. Loading it.".format(clnpath))
    cldict_noise = read_dict_from_hd5(clnpath)
else:
    print("Computing noise C_l's.")
    noisemap = cldict_normal_test['noisemaps'][0][0]
    cls_noise = hp.anafast([noisemap, noisemap, noisemap])
    labels = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
    cldict_noise = {labels[i]: cls_noise[i] for i in range(len(labels))}
    save_dict_to_hd5(clnpath, cldict_noise)
cldict_noise['ell'] = np.arange(len(cldict_noise['TT']), dtype='float')

prefix_noise = cldict_noise['ell']*(cldict_noise['ell'] + 1)/(2*np.pi)
print("Finished importing calibration_gain.")
print("="*80)

def myplot(cld, name, label, nf=1., bbmax=0.1, cldict_input=None,
           savefig=False):
    fig, ax = plt.subplots()
    ell = cld['ell']
    prefix = ell*(ell+1)/(2*np.pi)

    for bb in cld['BB']:
        ax.plot(ell, prefix*bb, 'k-', alpha=.015)

    bbmean = cld['BB_mean']
    bbstd = cld['BB_std']
    ax.errorbar(ell, prefix*bbmean, yerr=prefix*bbstd, label='Output')
    ax.plot(ell, 0*ell, 'g--', label='Input')

    # nmax = np.where(ell<20.)[0][-1]
    # bbmax = prefix[nmax]*(bbmean[nmax] + 5*bbstd[nmax])*2
    # bbmax = 0.1 # 0.007
    # ylim(ymax=bbmax)
    ax.set_ylim([0, bbmax])
    # xlim(xmin=2, xmax=20)
    ax.set_xlim(xmin=2, xmax=150)
    ax.set_xscale('log')

    ax.set_xlabel(r'$\ell$', fontsize=24)
    ax.set_ylabel(r'$D_\ell^{BB}$ ($\mu K^2$)', fontsize=24)

    if cldict_input is None:
        cldict_input = lib.cmb.get_cls(lensed=False,
                                       fname=datapath + 'r0p1_cl.dat')
    elif isinstance(cldict_input, dict):
        pass
    else:
        cldict_input = lib.cmb.get_cls(lensed=False, fname=cldict_input)
    ell0p1 = cldict_input['ell']
    bb0p1 = cldict_input['BB']
    prefix = ell0p1*(ell0p1 + 1)/(2*np.pi)

    # Plot base noise level
    prefix_noise = cldict_noise['ell']*(cldict_noise['ell'] + 1)/(2*np.pi)
    ax.plot(cldict_noise['ell'], prefix_noise*cldict_noise['BB']*nf**2, 'r-',
          label='Noise')

    # Plot weight-scaled noise level
    wQs = cld['weights_Q']
    wUs = cld['weights_U']
    wavg = (0.5*(wQs + wUs)).mean(axis=0)
    rnfact = cld['regnoise']/cld['regnoise'].min()
#     wfact = np.sum(wavg**2)
#     wfact = np.sum(np.abs(wavg)*rnfact)
#     wfact = np.sqrt(np.sum(wavg**2 * rnfact))
    wfact = np.sqrt(np.sum((wavg*rnfact)**2))*nf**2

    ax.plot(cldict_noise['ell'], prefix_noise*cldict_noise['BB']*wfact, 'y-',
          label='Weighted Noise')

    ax.plot(ell0p1, prefix*bb0p1, 'k--', label='r = 0.1')
    ax.legend(loc='upper left', fontsize=20)
    ax.set_title(label, fontsize=24)
    if savefig:
        np.savefig(name + '.png', dpi=150)
    # f = plt.gcf()
    return fig
