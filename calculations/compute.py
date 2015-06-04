#!/home/eswitzer/local/bin/python
import inspect
import os
import sys
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
               'N': 100}

argdicts = {}
arglist = [#('low_2_lownoise', [200., 270.], 0.1, '(200, 270) GHz, Low Noise'),
           ('all_4', [200., 270., 350., 600.], 1., 'All 4 Frequencies, Normal Noise'),
#           ('low_high_2', [200., 600.], 1., '(200, 600) GHz, Normal Noise'),
           ('all_4_lownoise', [200., 270., 350., 600.], 0.1, 'All 4 Frequencies, Low Noise'),
#           ('200_350', [200., 350.], 1., '(200, 350) GHz, Normal Noise'),
#           ('350_600', [350., 600.], 1., '(350, 600) GHz, Normal Noise')
           ]

tstart = time.time()

for i in arglist:
    name, freqs, noisefact, label = i
    regnoise = np.array([noisedict[float(freq)] for freq in freqs])
    freqs = np.array(freqs)*1.e9
    argdict = baseargdict.copy()
    argdict.update({'freqs': freqs, 'regnoise': regnoise*noisefact,
                    'noisefactor': noisefact, 'name': name})
    toadd = {'label': label, 'args': argdict}
    argdicts[name] = toadd

#results = {}
ells = np.unique(np.logspace(np.log10(2), np.log10(400), 200).astype('int'))
gains = [1.05]*len(ells)
for name, ad in argdicts.items():
#    results2 = {}
    t0 = time.time()
    print("Computing case: {0}".format(name))
    print("ells = {0}".format(ells))
    for i in range(len(ells)):
        ell = ells[i]
        gain = gains[i]
        fname = 'ell_{0}.hd5'.format(str(int(ell)))
        fname = os.path.abspath(datapath + '/' + name + '/' + fname)
#        print("fname = {0}, exists = {1}".format(fname, os.path.exists(fname))) #DELME
        if os.path.exists(fname):
            print("ell = {0} already exists at {1}. Skipping.".format(ell, fname))
        else:
            print("Starting ell = {0}".format(ell))
            t1 = time.time()
            cal_gains = [[ell], [gain]]

            cld = cg.many_realizations_parallel(cal_gains=cal_gains,
                                                **(ad['args']))
            cg.save_dict_to_hd5(fname, cld)
#            results2[ell] = {'cldict': cld, 'gain': gain}
            t2 = time.time()
            print("Finished ell = {0} in {1} seconds.".format(ell, t2-t1))
        print('.'*10)
    tf = time.time()
#    results[name] = results2
    print("Finished computing ({1} s): {0}".format(name, tf - t0))
    print('-'*30)

# print "Saving results to: ", calcpath + 'all_results.pickle'
# cg.save_data(results, calcpath + 'all_results.pickle')

tfinish = time.time()
print "-"*80
print "Finished computation in {0} seconds!".format(tfinish-tstart)

sys.exit(0)
