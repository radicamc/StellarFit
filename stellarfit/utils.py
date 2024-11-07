#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 12:44 2024

@author: MCR

Miscellaneous tools.
"""

from datetime import datetime
import h5py
import numpy as np
import os


def download_stellar_spectra(st_teff, st_logg, st_met, outdir, silent=False):
    """Download a grid of PHOENIX model stellar spectra.

    Parameters
    ----------
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].
    outdir : str
        Output directory.
    silent : bool
        If True, do not show any prints.

    Returns
    -------
    wfile : str
        Path to wavelength file.
    ffiles : list[str]
        Path to model stellar spectrum files.
    """

    fpath = 'ftp://phoenix.astro.physik.uni-goettingen.de/'

    # Get wavelength grid.
    wave_file = 'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    wfile = '{}/{}'.format(outdir, wave_file)
    if not os.path.exists(wfile):
        if not silent:
            fancyprint('Downloading file {}.'.format(wave_file))
        cmd = 'wget -q -O {0} {1}HiResFITS/{2}'.format(wfile, fpath, wave_file)
        os.system(cmd)
    else:
        if not silent:
            fancyprint('File {} already downloaded.'.format(wfile))

    # Get stellar spectrum grid points.
    teffs, loggs, mets = get_stellar_param_grid(st_teff, st_logg, st_met)

    # Construct filenames to retrieve
    ffiles = []
    for teff in teffs:
        for logg in loggs:
            for met in mets:
                if met > 0:
                    basename = 'lte0{0}-{1}0+{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                elif met == 0:
                    basename = 'lte0{0}-{1}0-{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                else:
                    basename = 'lte0{0}-{1}0{2}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'
                thisfile = basename.format(teff, logg, met)

                ffile = '{}/{}'.format(outdir, thisfile)
                ffiles.append(ffile)
                if not os.path.exists(ffile):
                    if not silent:
                        fancyprint('Downloading file {}.'.format(thisfile))
                    if met > 0:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z+{2}/{3}'.format(ffile, fpath, met, thisfile)
                    elif met == 0:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-{2}/{3}'.format(ffile, fpath, met, thisfile)
                    else:
                        cmd = 'wget -q -O {0} {1}HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z{2}/{3}'.format(ffile, fpath, met, thisfile)
                    os.system(cmd)
                else:
                    if not silent:
                        fancyprint('File {} already downloaded.'.format(ffile))

    return wfile, ffiles


def fancyprint(message, msg_type='INFO'):
    """Fancy printing statement mimicking logging.

    Parameters
    ----------
    message : str
        Message to print.
    msg_type : str
        Type of message. Mirrors the jwst pipeline logging.
    """

    time = datetime.now().isoformat(sep=' ', timespec='milliseconds')
    print('{} - StellarFit - {} - {}'.format(time, msg_type, message))


def get_param_dict_from_fit(filename, method='median', mcmc_burnin=None,
                            mcmc_thin=15, silent=False, drop_chains=None):
    """Reformat fit outputs from MCMC or NS into the parameter dictionary
    format expected by Model.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    method : str
        Method via which to get best fitting parameters from MCMC chains.
        Either "median" or "maxlike".
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain
        length. Only for MCMC.
    mcmc_thin : int
        Increment by which to thin chains. Only for MCMC.
    silent : bool
        If False, print messages.
    drop_chains : list(int), None
        Indices of chains to drop.

    Returns
    -------
    param_dict : dict
        Dictionary of light curve model parameters.
    """

    if not silent:
        fancyprint('Importing fitted parameters from file '
                   '{}.'.format(filename))

    # Get sample chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            chain = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(chain)[0])
            # Cut steps for burn in.
            chain = chain[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                chain = np.delete(chain, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(chain)
            # Flatten chains.
            chain = chain.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
            sampler = 'mcmc'
        elif 'ns' in list(f.keys()):
            chain = f['ns']['chain'][()]
            sampler = 'ns'
        else:
            msg = 'No MCMC or Nested Sampling results in file ' \
                  '{}.'.format(filename)
            raise KeyError(msg)
        # Either get maximum likelihood solution...
        if method == 'maxlike':
            if sampler == 'mcmc':
                lp = f['mcmc']['log_prob'][()].flatten()[mcmc_burnin:][::mcmc_thin]
                ii = np.argmax(lp)
                bestfit = chain[ii]
            else:
                bestfit = chain[-1]
        # ...or take median of samples.
        elif method == 'median':
            bestfit = np.nanmedian(chain, axis=0)

        # HDF5 groups are in alphabetical order. Reorder to match original
        # inputs.
        params, order = [], []
        for param in f['inputs'].keys():
            params.append(param)
            order.append(f['inputs'][param].attrs['location'])
        ii = np.argsort(order)
        params = np.array(params)[ii]

        # Create the parameter dictionary expected for Model using the fixed
        # parameters from the original inputs and the MCMC results.
        param_dict = {}
        pcounter = 0
        for param in params:
            param_dict[param] = {}
            dist = f['inputs'][param]['distribution'][()].decode()
            # Used input values for fixed parameters.
            if dist == 'fixed':
                param_dict[param]['value'] = f['inputs'][param]['value'][()]
            # Use fitted values for others.
            else:
                param_dict[param]['value'] = bestfit[pcounter]
                pcounter += 1

    return param_dict


def get_results_from_fit(filename, mcmc_burnin=None, mcmc_thin=15,
                         silent=False, drop_chains=None):
    """Extract posterior sample statistics (median and 1 sigma bounds) for
    each fitted parameter.

    Parameters
    ----------
    filename : str
        Path to file with MCMC fit outputs.
    mcmc_burnin : int
        Number of steps to discard as burn in. Defaults to 75% of chain
        length. Only for MCMC.
    mcmc_thin : int
        Increment by which to thin chains. Only for MCMC.
    silent : bool
        If False, print messages.
    drop_chains : list(int), None
        Indices of chains to drop.

    Returns
    -------
    results_dict : dict
        Dictionary of posterior medians and 1 sigma bounds for each fitted
        parameter.
    """

    if not silent:
        fancyprint('Importing fit results from file {}.'.format(filename))

    # Get MCMC chains from HDF5 file and extract best fitting parameters.
    with h5py.File(filename, 'r') as f:
        if 'mcmc' in list(f.keys()):
            chain = f['mcmc']['chain'][()]
            # Discard burn in and thin chains.
            if mcmc_burnin is None:
                mcmc_burnin = int(0.75 * np.shape(chain)[0])
            # Cut steps for burn in.
            chain = chain[mcmc_burnin:]
            # Drop chains if necessary.
            if drop_chains is not None:
                drop_chains = np.atleast_1d(drop_chains)
                chain = np.delete(chain, drop_chains, axis=1)
            nwalkers, nchains, ndim = np.shape(chain)
            # Flatten chains.
            chain = chain.reshape(nwalkers * nchains, ndim)[::mcmc_thin]
        elif 'ns' in list(f.keys()):
            chain = f['ns']['chain'][()]
        else:
            msg = 'No MCMC or Nested Sampling results in file ' \
                  '{}.'.format(filename)
            raise KeyError(msg)

        # HDF5 groups are in alphabetical order. Reorder to match original
        # inputs.
        params, order = [], []
        for param in f['inputs'].keys():
            params.append(param)
            order.append(f['inputs'][param].attrs['location'])
        ii = np.argsort(order)
        params = np.array(params)[ii]

        # Create the parameter dictionary expected for Model using the fixed
        # parameters from the original inputs and the MCMC results.
        results_dict = {}
        pcounter = 0
        for param in params:
            dist = f['inputs'][param]['distribution'][()].decode()
            # Skip fixed paramaters.
            if dist == 'fixed':
                continue
            # Get posterior median and 1 sigma range for fitted paramters.
            else:
                results_dict[param] = {}
                med = np.nanmedian(chain[:, pcounter], axis=0)
                low, up = np.diff(np.nanpercentile(chain[:, pcounter], [16, 50, 84]))
                results_dict[param]['median'] = med
                results_dict[param]['low_1sigma'] = low
                results_dict[param]['up_1sigma'] = up
                pcounter += 1

    return results_dict


def get_stellar_param_grid(st_teff, st_logg, st_met):
    """Given a set of stellar parameters, determine the neighbouring grid
    points based on the PHOENIX grid steps.

    Parameters
    ----------
    st_teff : float
        Stellar effective temperature.
    st_logg : float
        Stellar log surface gravity.
    st_met : float
        Stellar metallicity as [Fe/H].

    Returns
    -------
    teffs : list[float]
        Effective temperature grid bounds.
    loggs : list[float]
        Surface gravity grid bounds.
    mets : list[float]
        Metallicity grid bounds.
    """

    # Determine lower and upper teff steps (step size of 100K).
    teff_lw = int(np.floor(st_teff / 100) * 100)
    teff_up = int(np.ceil(st_teff / 100) * 100)
    if teff_lw == teff_up:
        teffs = [teff_lw]
    else:
        teffs = [teff_lw, teff_up]

    # Determine lower and upper logg step (step size of 0.5).
    logg_lw = np.floor(st_logg / 0.5) * 0.5
    logg_up = np.ceil(st_logg / 0.5) * 0.5
    if logg_lw == logg_up:
        loggs = [logg_lw]
    else:
        loggs = [logg_lw, logg_up]

    # Determine lower and upper metallicity steps (step size of 1).
    met_lw, met_up = np.floor(st_met), np.ceil(st_met)
    # Hack to stop met_up being -0.0 if -1<st_met<0.
    if -1 < st_met < 0:
        met_up = 0.0
    if met_lw == met_up:
        mets = [met_lw]
    else:
        mets = [met_lw, met_up]

    return teffs, loggs, mets
