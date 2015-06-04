#!/bin/python
"""
calibration_gain.py
jlazear
5/28/15

One-line description

Long description

Example:

<example code here>
"""
from __future__ import print_function

__version__ = 20150528
__releasestatus__ = 'beta'

import os
import inspect
import pickle
import time
import errno

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import h5py
from IPython import parallel as p
rc = p.Client()
cluster = rc[:]
cluster.block = True

import lib
import calculations.ilc_foreground_cleaning

datapath = os.path.abspath(os.path.dirname(os.path.abspath(inspect.getfile(
    lib))) + '/../data/') + '/'


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


def save_dict_to_hd5(f, d, replace=False, prefix=None, root=None, verbose=False):
    # Define the recursive worker function
    def _recurse(f, d):
        for key, value in d.items():
            if isinstance(value, dict):
                grp = f.create_group(key)
                _recurse(grp, value)
            else:
                f.create_dataset(key, data=value)

    # Make an hd5 file if we haven't passed one in
    if not isinstance(f, (h5py.File, h5py.Group)):
        if prefix is None:
            prefix = '' if f.startswith('/') else '.'
        fname = os.path.abspath(prefix + '/' + f)
        if verbose: print("Making hd5 file: {0}".format(fname))
        dirname = os.path.dirname(fname)
        mkdir_p(dirname)
        if replace:
            try:
                os.remove(fname)
            except OSError:
                pass
        with h5py.File(fname) as ff:
            if root is not None:
                rootdir = os.path.dirname(root + '/')
                rootdir = rootdir.lstrip('./')
                if rootdir:
                    ff = ff.create_group(root)
            _recurse(ff, d)
    else:
        if root is not None:
            rootdir = os.path.dirname(root + '/')
            rootdir = rootdir.lstrip('./')
            if rootdir:
                f = f.create_group(root)
        _recurse(f, d)


def read_dict_from_hd5(f):
    def _recurse(v):
        d = {}
        for key, value in v.items():
            if isinstance(value, h5py.Group):
                d[key] = _recurse(value)
            else:
                d[key] = value[:]
        return d        
    
    with h5py.File(f, 'r') as f:
        d = _recurse(f)
    return d


def make_dict_from_hd5_tree(directory, fname=None, verbose=True):
    matches = []
    for root, dirnames, filenames in os.walk('.'):
        print(root, dirnames, filenames)
        for filename in fnmatch.filter(filenames, '*.hd5'):
            matches.append(os.path.join(root, filename))

    if fname is None:
        fname = os.path.abspath(directory) + '.hd5'

    if verbose: print("Making hd5 file: {0}".format(fname))
    with h5py.File(fname) as f:
        for fname in matches:
            root = os.path.splitext(fname)[0]
            d = read_dict_from_hd5(fname)
            save_dict_to_hd5(f, d, root=root)


def many_realizations(freqs, N=100, xxs=['BB'], fname=None, regnoise=0., lensed=False,
                      modcov=False, verbose=False, **kwargs):
    print("Finished: {0} of {1}".format(0, N), end='\r')
    reconfunc = calculations.ilc_foreground_cleaning.polarized_ilc_reconstruction
    cldict = {key: [] for key in xxs}
    cldict['weights_Q'] = []
    cldict['weights_U'] = []
    for i in range(N):
        regendust = True if (i == 0) else False
        retdict = reconfunc(freqs, _debug=False, fname=fname, regnoise=regnoise,
                            lensed=lensed, verbose=verbose, modcov=modcov,
                            regeneratedust=regendust)
        cl_out = retdict['cl_out']
        for key in xxs:
            cldict[key].append(cl_out[key])
        cldict['weights_Q'].append(retdict['weights_Q'])
        cldict['weights_U'].append(retdict['weights_U'])

        print("Finished: {0} of {1}".format(i+1, N), end='\r')
    print("\n")

    for key, value in cldict.items():
        cldict[key] = np.array(value)
        cldict[key + '_mean'] = np.mean(cldict[key], axis=0)
        cldict[key + '_std'] = np.std(cldict[key], axis=0)

    cldict['ell'] = cl_out['ell']
    cldict['regnoise'] = retdict['regnoise']

    return cldict


def many_realizations_parallel(freqs, N=100, xxs=['BB'], fname=None,
                              regnoise=0., lensed=False,
                              modcov=False, verbose=False, name=None,
                              cal_gains=None, **kwargs):
    cldict = {key: [] for key in xxs}
    cldict['weights_Q'] = []
    cldict['weights_U'] = []

    # Generate dust maps
    if name is not None:
        dustname = datapath + name + '_dust.npy'
    else:
        dustname = None
    calculations.ilc_foreground_cleaning.regenerate_dust(freqs, fgfile=dustname)

    # Make and process N CMB realizations.
    def dummyfunc(n, freqs, _debug, fname, regnoise, lensed, verbose,
                  modcov, regeneratedust, cal_gains):
        reconfunc = calculations.ilc_foreground_cleaning.polarized_ilc_reconstruction
        temp = reconfunc(freqs, _debug=False, fname=fname,
                         regnoise=regnoise,  lensed=lensed,
                         verbose=verbose, modcov=modcov,
                         regeneratedust=False, cal_gains=cal_gains)
        toret = (temp['cl_out'], temp['weights_Q'], temp['weights_U'], temp['regnoise'])
        return toret
    print("Distributing {0} iterations to {1} cores.".format(N, len(cluster)))
    t0 = time.time()
    error = True
    while error:
        try:
            error = False
            retdicts = cluster.apply(dummyfunc, range(N), freqs, False,
                                     fname, regnoise, lensed, verbose,
                                     modcov, False, cal_gains)
        except p.CompositeError:
            error = True
            print("Error in worker process. Re-trying.")
    # IPython controller and view "conveniently" save results from computations in memory... 
    # This is a memory leak unless those saved results are discarded!
    rc.results.clear()
    cluster.results.clear()

    tf = time.time()
    print("Finished computation in {0} seconds".format(tf - t0))
    print("Collating results.")
    for (cl_out, weights_Q, weights_U, _) in retdicts:
        for key in xxs:
            cldict[key].append(cl_out[key])
        cldict['weights_Q'].append(weights_Q)
        cldict['weights_U'].append(weights_U)
#    for retdict in retdicts:
#        cl_out = retdict['cl_out']
#        for key in xxs:
#            cldict[key].append(cl_out[key])
#        cldict['weights_Q'].append(retdict['weights_Q'])
#        cldict['weights_U'].append(retdict['weights_U'])

    #     print("Finished: {0} of {1}".format(i+1, N), end='\r')
    # print("\n")

    for key, value in cldict.items():
        cldict[key] = np.array(value)
        cldict[key + '_mean'] = np.mean(cldict[key], axis=0)
        cldict[key + '_std'] = np.std(cldict[key], axis=0)

    cldict['ell'] = retdicts[0][0]['ell'] # cl_out['ell']  #DELME #FIXME
    cldict['regnoise'] = retdicts[0][3] # retdict['regnoise']  #DELME #FIXME

    return cldict


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

def myplot(cld, name, label, nf=1., bbmax=0.1, cldict_input=None):
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
    ax.title(label, fontsize=24)
    np.savefig(name + '.png', dpi=150)
    # f = plt.gcf()
    return fig


def save_data(results, fname='results.pickle'):
    copydict = results.copy()
#    for key in copydict.keys():
#        del copydict[key]['figure']
    with open(fname, 'w') as f:
        pickle.dump(copydict, f)
