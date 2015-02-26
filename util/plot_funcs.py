#!/bin/python

"""
plot_funcs
jlazear
2/26/15

Collection of useful plotting functions.
"""
__version__ = 20150226
__releasestatus__ = 'beta'


import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


def resid_plot(xs1, ys1, xs2, ys2, label1=None, label2=None,
               xlabel=None, ylabel=None, ratio=4, legend=True,
               fig=None, **kwargs):
    # Only include shared points for residual plot
    first = (len(xs1) <= len(xs2))
    xmax = len(xs1) if first else len(xs2)
    xssub = xs1 if first else xs2
    dys = ys2[:xmax] - ys1[:xmax]

    # Make fig and axes
    if fig is None:
        fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[ratio, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    plt.setp(ax1.get_xticklabels(), visible=False) # Remove ax1 x-tick labels

    # Plot data sets
    ax1.plot(xs1, ys1, label=label1, **kwargs)
    ax1.plot(xs2, ys2, label=label2, **kwargs)

    # Plot residuals
    ax2.plot(xssub, dys)

    # Set labels/legend
    ax2.set_xlabel(xlabel, fontsize=22)
    ax1.set_ylabel(ylabel, fontsize=22)
    ax2.set_ylabel('residuals', fontsize=16)
    if legend:
        ax1.legend(fontsize=16)

    # Adjust spacing
    fig.tight_layout(h_pad=0.0)

    return fig, [ax1, ax2]


def Dl_resid_plot(ells1, Cls1, ells2, Cls2, label1=None, label2=None,
                 xlabel=None, ylabel=None, ratio=4, legend=True, fig=None,
                 CltoDl=True, units=r'$\mathrm{\mu K^2}$', **kwargs):
    if xlabel is None:
        xlabel = r'$\ell \sim \frac{180^\circ}{\theta}$'
    if ylabel is None:
        if CltoDl:
            ylabel = r'$D_\ell^{XX}\,$' + r'({0})'.format(units)
        else:
            ylabel = r'$C_\ell^{XX}\,$' + r'({0})'.format(units)

    if CltoDl:
        Cls1 = (ells1*(ells1 + 1)/(2*np.pi))*Cls1
        Cls2 = (ells2*(ells2 + 1)/(2*np.pi))*Cls2

    return resid_plot(ells1, Cls1, ells2, Cls2,
                      label1=label1, label2=label2,
                      xlabel=xlabel, ylabel=ylabel,
                      ratio=ratio, legend=legend, fig=fig, **kwargs)


def print_summary(retdict):
    w_Q = retdict['weights_Q']
    w_U = retdict['weights_U']
    ilc_Q = retdict['ilcQmap']
    ilc_U = retdict['ilcUmap']
    try:
        debugdict = retdict['debug']
    except KeyError:
        print "No debug information available! Run ILC with _debug=True flag next time."
        print "frequencies = {0} GHz".format(retdict['frequencies'])
        print "weights_Q =", w_Q.flatten()
        print "weights_U =", w_U.flatten()
        print "var(ILC Qmap) =", ilc_Q.var()
        print "var(ILC Umap) =", ilc_U.var()
        return
    w_Qexp = debugdict['weights_Q']
    w_Uexp = debugdict['weights_U']
    Qcond = debugdict['G_Qcond']
    Ucond = debugdict['G_Ucond']
    ilc_Qexp = debugdict['ilcQmap']
    ilc_Uexp = debugdict['ilcUmap']

    print "frequencies = {0} GHz".format(retdict['frequencies']/1.e9)
    print "Q weights          =", w_Q.flatten()
    print "Q weights expected =", w_Qexp.flatten()
    print "log10(Q condition number) =", np.log10(Qcond)
    print "var(ILC Q)          =", ilc_Q.var()
    print "var(ILC Q) expected =", ilc_Qexp.var()
    print "U weights          =", w_U.flatten()
    print "U weights expected =", w_Uexp.flatten()
    print "log10(U condition number) =", np.log10(Ucond)
    print "var(ILC U)          =", ilc_U.var()
    print "var(ILC U) expected =", ilc_Uexp.var()