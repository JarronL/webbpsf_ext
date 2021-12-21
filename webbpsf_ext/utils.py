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
# Extend default wavelength range to 35 um
S.refs.set_default_waveset(minwave=500, maxwave=350000, num=10000.0, delta=None, log=False)
# JWST 25m^2 collecting area
# Flux loss from masks and occulters are taken into account in WebbPSF
# S.refs.setref(area = 25.4e4) # cm^2
S.refs.setref(area = 25.78e4) # cm^2 according to jwst_pupil_RevW_npix1024.fits.gz

# Progress bar
from tqdm.auto import trange, tqdm


def check_fitsgz(opd_file, inst_str=None):
    """
    WebbPSF FITS files can be either .fits or compressed .gz. 
    Search for .fits by default, then .fits.gz.

    Parameters
    ==========
    opd_file : str
        Name of FITS file, either .fits or .fits.gz
    inst_str : None or str
        If OPD file is instrument-specific, then specify here.
        Will look in instrument OPD directory. If set to None,
        then also checks `opd_file` for an instrument-specific
        string to determine if to look in instrument OPD directory,
        otherwise assume file name is in webbpsf data base directory.
    """
    from webbpsf.utils import get_webbpsf_data_path

    # Check if instrument name is in OPD file name
    # If so, then this is in instrument OPD directory
    # Otherwise, exists in webbpsf_data top directory

    if inst_str is None:
        inst_names = ['NIRCam', 'NIRISS', 'NIRSpec', 'MIRI', 'FGS']
        for iname in inst_names:
            if iname in opd_file:
                inst_str = iname

    # Get file directory
    if inst_str is None:
        # Location of JWST_OTE_OPD_RevAA_prelaunch_predicted.fits.gz
        opd_dir = get_webbpsf_data_path()
    else:
        opd_dir = os.path.join(get_webbpsf_data_path(),inst_str,'OPD')
    opd_fullpath = os.path.join(opd_dir, opd_file)

    # Check if file exists 
    # .fits or .fits.gz?
    if not os.path.exists(opd_fullpath):
        if '.gz' in opd_file:
            opd_file_alt = opd_file[:-3]
        else:
            opd_file_alt = opd_file + '.gz'
        opd_path_alt = os.path.join(opd_dir, opd_file_alt)
        if not os.path.exists(opd_path_alt):
            err_msg = f'Cannot find either {opd_file} or {opd_file_alt} in {opd_dir}'
            raise OSError(err_msg)
        else:
            opd_file = opd_file_alt

    return opd_file

