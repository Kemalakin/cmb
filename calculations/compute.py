import inspect
import os
import time

import numpy as np
import healpy as hp

import calculations.calibration_gain as cg
import lib

calcpath = os.path.abspath(os.path.dirname(os.path.abspath(inspect.getfile(
    lib))) + '/../calculations/') + '/'
datapath = os.path.abspath(os.path.dirname(os.path.abspath(inspect.getfile(
    lib))) + '/../data/') + '/'

Omega_single = 7.45446e-6  # sr, from bjohnson report
Omega_pixel = hp.nside2pixarea(512)  # pixel solid angle in sr
noiselist = 3./1.3*np.array([1.3, 1.9, 6.7, 110.])*np.sqrt(Omega_single/Omega_pixel)  # uK/pixel
noisedict = dict(zip([200., 270., 350., 600.], noiselist))
# regnoise = 3.*np.sqrt(Omega_single/Omega_pixel)  # uK/pixel
fname = datapath + 'explanatory_cl.dat'
lensed = False    # Include lensing?
noisefactor = 1.  # Noise level relative to PIPER target
modcov = False    # Efstathiou 2009 modified covariance?
verbose = False

baseargdict = {'lensed': lensed,
               'noisefactor': noisefactor,
               'modcov': modcov,
               'fname': fname,
               'verbose': verbose,
               'N': 2}

argdicts = {}
arglist = [('low_2_lownoise', [200., 270.], 0.1, '(200, 270) GHz, Low Noise'),
           #('all_4', [200., 270., 350., 600.], 1., 'All 4 Frequencies, Normal Noise'),
           #('low_high_2', [200., 600.], 1., '(200, 600) GHz, Normal Noise'),
           #('all_4_lownoise', [200., 270., 350., 600.], 0.1, 'All 4 Frequencies, Low Noise'),
           #('200_350', [200., 350.], 1., '(200, 350) GHz, Normal Noise'),
           #('350_600', [350., 600.], 1., '(350, 600) GHz, Normal Noise')
           ]

for i in arglist:
    name, freqs, noisefact, label = i
    regnoise = np.array([noisedict[float(freq)] for freq in freqs])
    freqs = np.array(freqs)*1.e9
    argdict = baseargdict.copy()
    argdict.update({'freqs': freqs, 'regnoise': regnoise*noisefact, 'noisefactor': noisefact})
    toadd = {'label': label, 'args': argdict}
    argdicts[name] = toadd

results = {}

for name, ad in argdicts.items():
    t0 = time.time()
    print("Computing case: {0}".format(name))
    cld = cg.many_realizations(**(ad['args']))
    results[name] = {'cldict': cld}
    # f = myplot(cld, name, ad['label'])
    # results[name]['figure'] = f
    tf = time.time()
    print("Finished computing ({1} s): {0}".format(name, tf - t0))
    print('-'*30)

print "Saving results to: ", calcpath + 'results.pickle'
cg.save_data(results, calcpath + 'results.pickle')
