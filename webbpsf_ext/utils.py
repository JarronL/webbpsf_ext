# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os, sys
import six

import webbpsf, poppy, pysiaf

on_rtd = os.environ.get('READTHEDOCS') == 'True'
# Preferred matplotlib settings
rcvals = {'xtick.minor.visible': True, 'ytick.minor.visible': True,
          'xtick.direction': 'in', 'ytick.direction': 'in',
          'xtick.top': True, 'ytick.right': True, 'font.family': ['serif'],
          'xtick.major.size': 6, 'ytick.major.size': 6,
          'xtick.minor.size': 3, 'ytick.minor.size': 3,
          'image.interpolation': 'nearest', 'image.origin': 'lower',
          'figure.figsize': [8,6], 'mathtext.fontset':'cm'}#,
          #'text.usetex': True, 'text.latex.preamble': ['\usepackage{gensymb}']}
if not on_rtd:
    matplotlib.rcParams.update(rcvals)
    cmap_pri, cmap_alt = ('viridis', 'gist_heat')
    matplotlib.rcParams['image.cmap'] = cmap_pri if cmap_pri in plt.colormaps() else cmap_alt

from . import conf
from .logging_utils import setup_logging

import logging
_log = logging.getLogger('webbpsf_ext')

import pysynphot as S
# Extend default wavelength range to 5.6 um
S.refs.set_default_waveset(minwave=500, maxwave=56000, num=10000.0, delta=None, log=False)
# JWST 25m^2 collecting area
# Flux loss from masks and occulters are taken into account in WebbPSF
# S.refs.setref(area = 25.4e4) # cm^2
S.refs.setref(area = 25.78e4) # cm^2 according to jwst_pupil_RevW_npix1024.fits.gz

# Progress bar
from tqdm.auto import trange, tqdm

# The following won't work on readthedocs compilation
if not on_rtd:
    # Grab WebbPSF assumed pixel scales
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    nc_temp = webbpsf.NIRCam()
    setup_logging(log_prev, verbose=False)

    pixscale_SW = nc_temp._pixelscale_short
    pixscale_LW = nc_temp._pixelscale_long
    del nc_temp

    # _jbt_exists = True
    # try:
    #     from jwst_backgrounds import jbt
    # except ImportError:
    #     _log.info("  jwst_backgrounds is not installed and will not be used for bg estimates.")
    #     _jbt_exists = False
