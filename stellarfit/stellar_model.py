#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Nov 07 12:35 2024

@author: MCR

Code to interpolate stellar models of inhomogeneous surfaces.
"""

from astropy.io import fits
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import spectres
from tqdm import tqdm

from stellarfit import utils


class StellarModel:

    def __init__(self, input_parameters, wavelengths, stellar_grid,
                 dt_min=100):
        self.stellar_grid = stellar_grid
        self.input_parameters = input_parameters
        self.spots, self.faculae = False, False
        self.wave = wavelengths
        self.model = None
        for key in self.input_parameters.keys():
            key_split = key.split('_')
            if len(key_split) > 1:
                if key_split[1] == 'spot':
                    self.spots = True
                elif key_split[1] == 'fac':
                    self.faculae = True

        for param in ['teff', 'logg_phot']:
            assert param in self.input_parameters.keys()
        if self.spots is True:
            for param in ['dt_spot', 'f_spot']:
                assert param in self.input_parameters.keys()
                if self.input_parameters['dt_spot']['value'] < dt_min:
                    msg = 'dt_spot less than minimum temperature contrast of {}K.'.format(
                        dt_min)
                    raise ValueError(msg)
        if self.faculae is True:
            for param in ['dt_fac', 'f_fac']:
                assert param in self.input_parameters.keys()
                if self.input_parameters['dt_fac']['value'] < dt_min:
                    msg = 'dt_fac less than minimum temperature contrast of {}K.'.format(
                        dt_min)
                    raise ValueError(msg)

        if 'scale' not in self.input_parameters.keys():
            self.input_parameters['scale'] = {}
            self.input_parameters['scale']['value'] = 1

    def compute_model(self):
        t, g = self.input_parameters['teff']['value'], \
               self.input_parameters['logg_phot']['value']
        phot_mod = self.stellar_grid((t, g))

        if self.spots is True:
            t_spot = t - self.input_parameters['dt_spot']['value']
            f_spot = self.input_parameters['f_spot']['value']
            if 'logg_spot' in self.input_parameters.keys():
                g_spot = self.input_parameters['logg_spot']['value']
            else:
                g_spot = g
            spot_mod = f_spot * self.stellar_grid((t_spot, g_spot))
        else:
            f_spot = 0
            spot_mod = np.zeros_like(phot_mod)

        if self.faculae is True:
            t_fac = t + self.input_parameters['dt_fac']['value']
            f_fac = self.input_parameters['f_fac']['value']
            if 'logg_fac' in self.input_parameters.keys():
                g_fac = self.input_parameters['logg_fac']['value']
            else:
                g_fac = g
            fac_mod = f_fac * self.stellar_grid((t_fac, g_fac))
        else:
            f_fac = 0
            fac_mod = np.zeros_like(phot_mod)

        if f_fac + f_spot >= 0.5:
            msg = 'Combined spot and facula covering fraction is >50%.'
            raise ValueError(msg)

        scale = self.input_parameters['scale']['value']
        star = scale * ((1 - f_spot - f_fac) * phot_mod + spot_mod + fac_mod)

        self.model = star


def load_phoenix_grid(temperatures, log_gs, data_wave, input_dir,
                      flux_conv_factor, silent=False):
    # PHOENIX Grid
    temperatures, log_gs = np.array(temperatures), np.array(log_gs)
    if np.max(temperatures) > 7000 or np.min(temperatures) < 2300:
        msg = 'Temperatures for the PHOENIX grid must be between 2300K and 7000K.'
        raise ValueError(msg)
    if np.max(log_gs) > 5.5 or np.min(log_gs) < 1.0:
        msg = 'log g values for the PHOENIX grid must be between 1.0 and 5.5.'
        raise ValueError(msg)

    print(
        'Loading PHOENIX model grid for temperatures in range {0} -- {1}K and log gravity in range {2} -- {3}.'.format(
            temperatures[0], temperatures[1], log_gs[0], log_gs[1]))

    t_steps = int((temperatures[1] - temperatures[0]) / 100 + 1)
    g_steps = int((log_gs[1] - log_gs[0]) / 0.5 + 1)
    temperatures = np.linspace(temperatures[0], temperatures[1],
                               t_steps).astype(int)
    log_gs = np.linspace(log_gs[0], log_gs[1], g_steps)

    spectra = []
    g_array, t_array = np.meshgrid(log_gs, temperatures)
    g_array = g_array.flatten()
    t_array = t_array.flatten()

    for i in tqdm(range(len(t_array))):
        temp, logg = t_array[i], g_array[i]
        res = utils.download_stellar_spectra(temp, logg, 0.0, input_dir,
                                             silent=silent)
        wave_file, flux_file = res

        mod_wave = fits.getdata(wave_file) / 1e4
        mod_spec = fits.getdata(flux_file[0]) * 1e-4 * flux_conv_factor
        mod_spec = spectres.spectres(data_wave, mod_wave, mod_spec)
        spectra.append(mod_spec)

    spectra = np.array([spectra])[0]
    spectra = np.reshape(spectra,
                         (len(temperatures), len(log_gs), len(spectra[0])))

    stellar_grid = RegularGridInterpolator(points=[temperatures, log_gs],
                                           values=spectra)

    return stellar_grid


def load_sphinx_grid(temperatures, log_gs, data_wave, input_dir,
                     flux_conv_factor):
    # SPHINX Grid
    temperatures, log_gs = np.array(temperatures), np.array(log_gs)
    if np.max(temperatures) > 4000 or np.min(temperatures) < 2000:
        msg = 'Temperatures for the SPHINX grid must be between 2000K and 4000K.'
        raise ValueError(msg)
    if np.max(log_gs) > 5.5 or np.min(log_gs) < 4.0:
        msg = 'log g values for the SPHINX grid must be between 4.0 and 5.5.'
        raise ValueError(msg)

    print(
        'Loading SPHINX model grid for temperatures in range {0} -- {1}K and log gravity in range {2} -- {3}.'.format(
            temperatures[0], temperatures[1], log_gs[0], log_gs[1]))

    t_steps = int((temperatures[1] - temperatures[0]) / 100 + 1)
    g_steps = int((log_gs[1] - log_gs[0]) / 0.25 + 1)
    temperatures = np.linspace(temperatures[0], temperatures[1],
                               t_steps).astype(int)
    log_gs = np.linspace(log_gs[0], log_gs[1], g_steps)

    spectra = []
    g_array, t_array = np.meshgrid(log_gs, temperatures)
    g_array = g_array.flatten()
    t_array = t_array.flatten()

    for i in tqdm(range(len(t_array))):
        temp, logg = t_array[i], g_array[i]
        file = 'Teff_{0}.0_logg_{1}_logZ_+0.0_CtoO_0.5.txt'.format(temp, logg)
        mod = pd.read_csv(input_dir + file, comment='#',
                          names=['wave', 'spec'], sep='\s+')
        mod['spec'] *= 1e-3
        mod['spec'] *= flux_conv_factor
        mod_spec = np.interp(data_wave, mod['wave'], mod['spec'])
        spectra.append(mod_spec)

    spectra = np.array([spectra])[0]
    spectra = np.reshape(spectra,
                         (len(temperatures), len(log_gs), len(spectra[0])))

    stellar_grid = RegularGridInterpolator(points=[temperatures, log_gs],
                                           values=spectra)

    return stellar_grid


def load_stellar_grid(temperatures, log_gs, data_wave, input_dir,
                      flux_conv_factor, model_type='PHOENIX', silent=False):
    if model_type == 'PHOENIX':
        stellar_grid = load_phoenix_grid(temperatures, log_gs, data_wave,
                                         input_dir, flux_conv_factor,
                                         silent=silent)

    elif model_type == 'SPHINX':
        stellar_grid = load_sphinx_grid(temperatures, log_gs, data_wave,
                                        input_dir, flux_conv_factor)
    else:
        raise ValueError('Unrecognized model type {}'.format(model_type))

    return stellar_grid
