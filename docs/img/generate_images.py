#!/bin/python

"""
generate_images.py
jlazear
1/22/15

Generates images used in the docs files.

Example:

$ python generate_images.py
"""
__version__ = 20150122
__releasestatus__ = 'beta'


import os
import sys
import argparse

import wget
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

imgpath = os.path.dirname(os.path.abspath(__file__)) + '/'
cmbpath = os.path.abspath(imgpath + '/../../') + '/'
sys.path.append(cmbpath)
import lib


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--regenerate",
                    help="force the regeneration of all figures",
                    action="store_true")
args = parser.parse_args()
force = bool(args.regenerate)

strtemplate = "Saving {0} to {1}\r\n"
strexists = "File {0} already exists."

# -------------------------------------
# ---------- Generate images ----------
# -------------------------------------

# planck_dust_353GHz.png
fname = imgpath + 'planck_dust_353GHz.png'
if os.path.isfile(fname) and not force:
    print strexists.format(fname)
else:
    dust353map = lib.foregrounds.generate_simple_dust_map()
    fig = plt.figure(1)
    hp.mollview(dust353map, fig=1, nest=True, norm='hist', unit='MJy/sr',
                title='')

    print strtemplate.format('Planck 353 GHz dust map', fname)
    fig.savefig(fname)
    fig.clf()


# cb9_f22_M.png
cb9_f22_fname = imgpath + 'cb9_f22_M.png'
if os.path.isfile(cb9_f22_fname) and not force:
    print strexists.format(cb9_f22_fname)
else:
    try:
        os.remove(cb9_f22_fname)
    except OSError:
        pass
    cb9_f22_url = ('http://lambda.gsfc.nasa.gov/product/map/current/pub_papers'
                   '/nineyear/basic_results/images/med/cb9_f22_M.png')
    print strtemplate.format('CB9 Fig22', cb9_f22_fname)
    wget.download(cb9_f22_url, out=cb9_f22_fname)
    print


# pol_dust.png
fname = imgpath + 'pol_dust_353GHz.png'
if os.path.isfile(fname) and not force:
    print strexists.format(fname)
else:
    dust353map = lib.foregrounds.generate_simple_dust_map()
    pfrac = 0.2
    poldustmap = pfrac*dust353map
    fig = plt.figure(1)
    hp.mollview(poldustmap, fig=1, nest=True, norm='hist', unit='MJy/sr', title='')

    print strtemplate.format('Simple polarized dust map', fname)
    fig.savefig(fname)
    fig.clf()


# QU.png
fnameQ = imgpath + 'Q.png'
fnameU = imgpath + 'U.png'
if os.path.isfile(fnameQ) and os.path.isfile(fnameU) and not force:
    print strexists.format(fnameQ)
    print strexists.format(fnameU)
else:
    dust353map = lib.foregrounds.generate_simple_dust_map()
    pfrac = 0.2
    poldustmap = pfrac*dust353map
    thetamap = np.random.rand(len(dust353map))*2*np.pi  # [0, 1)*2pi = [0, 2pi)
    Qfact = np.cos(thetamap)
    Ufact = np.sin(thetamap)
    Qmap = poldustmap*Qfact
    Umap = poldustmap*Ufact

    fig = plt.figure(1)
    hp.mollview(Qmap, fig=1, nest=True, norm='hist', unit='MJy/sr', title='')
    print strtemplate.format('Simple Q map', fnameQ)
    fig.savefig(fnameQ)
    fig.clf()

    fig = plt.figure(1)
    hp.mollview(Umap, fig=1, nest=True, norm='hist', unit='MJy/sr', title='')
    print strtemplate.format('Simple U map', fnameU)
    fig.savefig(fnameU)
    fig.clf()


#gammaQU
fnames = [imgpath + 'gamma.png', imgpath + 'gammaQ.png', imgpath + 'gammaU.png']
allthere = all(map(os.path.isfile, fnames))
if allthere and not force:
    for fname in fnames:
        print strexists.format(fname)
else:
    # dust353map = lib.foregrounds.generate_simple_dust_map()
    # pfrac = 0.2
    # poldustmap = pfrac*dust353map
    # mapnames = ['wmap_mcmc_k_synch_stk_q_7yr_v4p1.fits',
    #             'data/wmap_mcmc_k_synch_stk_u_7yr_v4p1.fits']
    # wmapQ, wmapU = lib.wmap.load_wmap_maps_QU(mapnames, Nside=256, n2r=True)
    # gamma = lib.foregrounds.generate_polarization_angle_map(wmapQ, wmapU,
    #                                                         sigma=3.*np.pi/180)
    # Qmap = poldustmap*np.cos(2*gamma)
    # Umap = poldustmap*np.sin(2*gamma)

    QUmaps = lib.foregrounds.generate_synchro_traced_dust_QU_map(
        sigma=3.*np.pi/180)
    gamma = 0.5*np.arctan2(-QUmaps[0], QUmaps[1])

    fnamegamma, fnameQ, fnameU = fnames
    fig = plt.figure(1)
    hp.mollview(gamma, fig=1, nest=True, unit='radians', title='')
    print strtemplate.format('Pol Dust Angle map', fnamegamma)
    fig.savefig(fnamegamma)
    fig.clf()

    fig = plt.figure(1)
    hp.mollview(QUmaps[0], fig=1, nest=True, unit='MJy/sr', norm='hist',
                title='')
    print strtemplate.format('Q map from synch angle', fnameQ)
    fig.savefig(fnameQ)
    fig.clf()

    fig = plt.figure(1)
    hp.mollview(QUmaps[1], fig=1, nest=True, unit='MJy/sr', norm='hist',
                title='')
    print strtemplate.format('U map from synch angle', fnameU)
    fig.savefig(fnameU)
    fig.clf()

#noise_table
fname = imgpath + 'noise_table.txt'
if os.path.isfile(fname) and not force:
    print strexists.format(fname)
else:
    NEPs_photon = np.array([7., 8., 11., 13.])*1.e-18  # W/sqrt(Hz)
    NEPs_phonon = np.array([4., 4., 4., 4.])*1.e-18  # W/sqrt(Hz)
    etas = np.array([0.9, 0.9, 0.7, 0.5])  # unitless detector absorption efficiency
    taus = np.array([.55, .52, .5, .42])  # unitless optical efficiency
    nu_mids = np.array([200., 270., 350., 600.])*1.e9   # Band center frequency in GHz
    bws = np.array([.3, .3, .16, .1])  # Band width dnu/nu
    nu_mins = nu_mids*(1 - bws/2.)  # Lower cut-off of frequency bands
    nu_maxs = nu_mids*(1 + bws/2.)  # Upper cut-off of frequency bands

    fQ = fU = 4./17
    fV = 13./17

    fnum = 1.6  # At detector
    Omega = 4*np.pi/(4*fnum)**2  # At detector
    A_single = (1135.e-6)**2  # 1135 um x 1135 um pixels

    dPdTs = [lib.noise.dPdT(nu_mins[i], nu_maxs[i], eta_optical=taus[i], eta_detector=etas[i],
                            A=A_single, Omega=Omega, verbose=False) for i in range(4)]

    NEQ_phonons = NEU_phonons = np.sqrt(2/fQ)*1/np.sqrt(taus*etas)*NEPs_phonon/dPdTs  # K sqrt(s)
    NEV_phonons = np.sqrt(2/fV)*1/np.sqrt(taus*etas)*NEPs_phonon/dPdTs  # K sqrt(s)

    NEQ_photons = NEU_photons = np.sqrt(2/fQ)*1/np.sqrt(taus*etas)*NEPs_photon/dPdTs  # K sqrt(s)
    NEV_photons = np.sqrt(2/fV)*1/np.sqrt(taus*etas)*NEPs_photon/dPdTs  # K sqrt(s)

    linestr = "{0:^30} {1:^10} {2:^10} {3:^10} {4:^10}"
    linebreak = ('-'*30, np.array(['-'*10]*4))
    fmtstr = lambda x: ('{0:.3e}' if len(str(x)) > 10 else '{0}').format(x)

    elemlist = [('Frequency (GHz)', ['200', '270', '350', '600']),
                ('Bandwidth (dnu/nu)', bws),
                linebreak,
                ('eta', etas),
                ('tau', taus),
                linebreak,
                ('Single Pixel Area (m^2)', A_single + bws*0),
                ('f-number (f/N)', fnum + bws*0),
                ('Single Pixel Omega (sr)', Omega + bws*0),
                ('AOmega (m^2 sr)', A_single*Omega + bws*0),
                linebreak,
                ('f_Q/U', fQ + bws*0),
                ('f_V', fV + bws*0),
                linebreak,
                ('NEP (photon) (W/sqrt(Hz))', NEPs_photon),
                ('NEP (phonon) (W/sqrt(Hz))', NEPs_phonon),
                linebreak,
                ('dP/dTs (W/K)', dPdTs),
                ('NEQ_phonons (uK sqrt(s))', NEQ_phonons*1.e6),
                ('NEU_phonons (uK sqrt(s))', NEU_phonons*1.e6),
                ('NEV_phonons (uK sqrt(s))', NEV_phonons*1.e6)]

    t0 = 10.*3600/(0.5*4*np.pi)  # 10 hours/half sky -> s/sr
    farrays = 4.  # 4 co-pointed arrays

    nind = np.array([943., 1550., 2270., 3760.])
    fs = np.sqrt(nind/(4.*32*40))  # fraction of non-overlapping beams
    frows_photon = fs*32.  # number of independent pixels (beamspots) along the column (polar direction)
    frows_phonon = 32. # number of independent pixels for phonons

    fscan = 1.5  # scan strategy overlap

    t_photon = t0*farrays*frows_photon*fscan  # s/sr
    t_phonon = t0*farrays*frows_phonon*fscan + t_photon*0  # s/sr

    mQ_photons = mU_photons = 1/np.sqrt(t_photon)*NEQ_photons
    mV_photons = 1/np.sqrt(t_photon)*NEV_photons

    mQ_phonons = mU_phonons = 1/np.sqrt(t_phonon)*NEQ_phonons
    mV_phonons = 1/np.sqrt(t_phonon)*NEV_phonons

    mQ_total = np.sqrt(mQ_photons**2 + mQ_phonons**2)
    mU_total = np.sqrt(mU_photons**2 + mU_phonons**2)
    mV_total = np.sqrt(mV_photons**2 + mV_phonons**2)

    # theta_beam = 19./60*np.pi/180.  # radians
    theta_beam = 6.*np.pi/180.  # radians
    Omega_beam = 2*np.pi*(1 - np.cos(theta_beam/2.))  # sr
    Omega_pixel = hp.nside2pixarea(512)  # sr

    linestr = "{0:^30} {1:^10} {2:^10} {3:^10} {4:^10}"
    linebreak = ('-'*30, np.array(['-'*10]*4))
    fmtstr = lambda x: ('{0:.3e}' if (len(str(x)) > 10) else '{0}').format(x)[:10]

    elemlist2 = [linebreak,
                 ('t (phonon) (10^6 s/sr)', t_phonon/1.e6),
                 ('t (photon) (10^6 s/sr)', t_photon/1.e6),
                 linebreak,
                 ('mQ (phonons) (uK sqrt(sr))', mQ_phonons*1.e6),
                 ('mU (phonons) (uK sqrt(sr))', mU_phonons*1.e6),
                 ('mV (phonons) (uK sqrt(sr))', mV_phonons*1.e6),
                 ('mQ (photons) (uK sqrt(sr))', mQ_photons*1.e6),
                 ('mU (photons) (uK sqrt(sr))', mU_photons*1.e6),
                 ('mV (photons) (uK sqrt(sr))', mV_photons*1.e6),
                 ('mQ (total) (uK sqrt(sr))', mQ_total*1.e6),
                 ('mU (total) (uK sqrt(sr))', mU_total*1.e6),
                 ('mV (total) (uK sqrt(sr))', mV_total*1.e6),
                 linebreak,
                 ('dT_Q (beamspot) (uK)', mQ_total*1.e6/np.sqrt(Omega_beam)),
                 ('dT_U (beamspot) (uK)', mU_total*1.e6/np.sqrt(Omega_beam)),
                 ('dT_V (beamspot) (uK)', mV_total*1.e6/np.sqrt(Omega_beam)),
                 linebreak,
                 ('dT_Q (pixel) (uK)', mQ_total*1.e6/np.sqrt(Omega_pixel)),
                 ('dT_U (pixel) (uK)', mU_total*1.e6/np.sqrt(Omega_pixel)),
                 ('dT_V (pixel) (uK)', mV_total*1.e6/np.sqrt(Omega_pixel))]

    elemlist += elemlist2

    with open(fname, 'w') as f:
        for key, value in elemlist:
            values = [fmtstr(i) for i in value]
            f.write(linestr.format(key, *values) + '\r\n')
