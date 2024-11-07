#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 13:04 2024

@author: MCR

Plotting functions.
"""

import corner
import h5py
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import numpy as np


def make_corner_plot(filename, mcmc_burnin=None, mcmc_thin=15, labels=None,
                     outpdf=None, log_params=None, drop_chains=None):
    """Make a corner plot of fitted posterior distributions.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain
        length. MCMC only.
    mcmc_thin : int
        Increment by which to thin chains. MCMC only.
    labels : list(str)
        Fitted parameter names.
    outpdf : PdfPages
        Path to file to save plot.
    log_params : list(int), None
        Indices of parameters to show on log scale.
    drop_chains : list(int), None
        Indices of chains to drop.
    """

    # Get chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            samples = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(samples)[0])
            # Cut steps for burn in.
            samples = samples[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                samples = np.delete(samples, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(samples)
            # Flatten chains.
            samples = samples.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
        elif 'ns' in list(f.keys()):
            samples = f['ns']['chain'][()]

    # Log certain parameters if necessary.
    if log_params is not None:
        log_params = np.atleast_1d(log_params)
        for i in log_params:
            samples[:, i] = np.log10(samples[:, i])

    # Make corner plot
    figure = corner.corner(samples, labels=labels, show_titles=True)

    if outpdf is not None:
        if isinstance(outpdf, matplotlib.backends.backend_pdf.PdfPages):
            outpdf.savefig(figure)
        else:
            figure.savefig(outpdf)
        figure.clear()
        plt.close(figure)
    else:
        plt.show()


def plot_mcmc_chains(filename, labels=None, log_params=None,
                     highlight_chains=None, drop_chains=None):
    """Plot MCMC chains.

    Parameters
    ----------
    filename : str
        MCMC output file.
    labels : list(str)
        Fitted parameter names.
    log_params : list(str), None
        Indices of parameters to plot in log-space.
    highlight_chains : list(int), None
        Indices of chains to highlight.
    drop_chains : list(int), None
        Indices of chains to drop.
    """

    # Get MCMC chains.
    with h5py.File(filename, 'r') as f:
        samples = f['mcmc']['chain'][()]

    # Convert to log-scale if necessary.
    if log_params is not None:
        log_params = np.atleast_1d(log_params)
        for i in log_params:
            samples[:, :, i] = np.log10(samples[:, :, i])

    # Drop chains if necessary.
    if drop_chains is not None:
        drop_chains = np.atleast_1d(drop_chains)
        samples = np.delete(samples, drop_chains, axis=1)

    nwalkers, nchains, ndim = np.shape(samples)
    # Plot chains.
    fig, axes = plt.subplots(ndim,
                             figsize=(10, np.ceil(ndim / 1.25).astype(int)),
                             sharex=True)

    if highlight_chains is not None:
        highlight_chains = np.atleast_1d(highlight_chains)

    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], c='black', alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.yaxis.set_label_coords(-0.1, 0.5)
        if labels is not None:
            ax.set_ylabel(labels[i])
        if highlight_chains is not None:
            for j in highlight_chains:
                ax.plot(samples[:, j, i], c='red', alpha=0.5)

    axes[-1].set_xlabel('Step Number')
    plt.show()
