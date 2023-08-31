# Import libraries
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os, sys
import six

import webbpsf, poppy, pysiaf

# Define these here rather than calling multiple times
# since it takes some time to generate these.
siaf_nrc = pysiaf.Siaf('NIRCam')
siaf_nis = pysiaf.Siaf('NIRISS')
siaf_mir = pysiaf.Siaf('MIRI')
siaf_nrs = pysiaf.Siaf('NIRSpec')
siaf_fgs = pysiaf.Siaf('FGS')

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
        # Location of JWST_OTE_OPD_*.fits.gz
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

def get_one_siaf(filename=None, instrument='NIRCam'):
    """
    Convenience function to import a SIAF XML file to override the
    default pysiaf aperture information.

    Parameters
    ==========
    filename : str or None
        Name of SIAF file (e.g., 'NIRCam_SIAF.xml').
        If not set, returns default SIAF object.
    instrument : str
        Name of instrument associated with XML file. 
    """
    si_match = {
        'NIRCAM' : siaf_nrc, 
        'NIRSPEC': siaf_nis, 
        'MIRI'   : siaf_mir, 
        'NIRISS' : siaf_nrs, 
        'FGS'    : siaf_fgs,
        }

    siaf_object = deepcopy(si_match[instrument.upper()])

    if filename is None:
        return siaf_object
    else:
        aperture_collection_NRC_base = pysiaf.read.read_jwst_siaf(filename=filename)
        siaf_object.apertures = aperture_collection_NRC_base
        siaf_object.description = os.path.basename(filename)
        siaf_object.observatory = 'JWST'
        return siaf_object
    

def get_detname(det_id, use_long=False):
    """Return NRC[A-B][1-4,5/LONG] for valid detector/SCA IDs
    
    Parameters
    ==========
    det_id : int or str
        Detector ID, either integer SCA ID or string detector name.
    use_long : bool
        For longwave detectors, return 'LONG' instead of '5' in detector name.
    """

    # For longwave devices, do we use 'LONG' or '5'?
    long_str_use = 'LONG' if use_long else '5'
    long_str_not = '5' if use_long else 'LONG'


    det_dict = {481:'A1', 482:'A2', 483:'A3', 484:'A4', 485:f'A{long_str_use}',
                486:'B1', 487:'B2', 488:'B3', 489:'B4', 490:f'B{long_str_use}'}
    scaids = det_dict.keys()
    detids = det_dict.values()
    detnames = ['NRC' + idval for idval in detids]

    # If already valid, then return
    if det_id in detnames:
        return det_id
    elif det_id in scaids:
        detname = 'NRC' + det_dict[det_id]
    elif det_id.upper() in detids:
        detname = 'NRC' + det_id.upper()
    else:
        detname = det_id

    # If NRCA5 or NRCB5, change '5' to 'LONG' 
    detname = detname.upper()
    if long_str_not in detname:
        detname = detname.replace(long_str_not, long_str_use)
        # Ensure NRC is prepended
        if detname[0:3]!='NRC':
            detname = 'NRC' + detname

    if detname not in detnames:
        all_names = ', '.join(detnames)
        err_str = f"Invalid detector: {detname} \n\tValid names are: {all_names}"
        raise ValueError(err_str)
        
    return detname

def pix_ang_size(ap, sr=True, pixscale=None):
    """Angular area of pixel from aperture object
    
    If `sr=True` then return in sterradians,
    otherwise return in units of arcsec^2.
    """
    sr2asec2 = 42545170296.1522
    
    # X and Y scale in arcsec / pixel
    if pixscale is not None:
        if isinstance(pixscale, (np.ndarray, list, tuple)):
            xscale, yscale = pixscale
        else:
            xscale = yscale = pixscale
    else:
        xscale = ap.XSciScale
        yscale = ap.YSciScale
    
    # Area in sq arcsec
    area_asec2 = xscale * yscale
    
    if sr:
        # Convert to sterradian
        return area_asec2 / sr2asec2
    else:
        return area_asec2
    