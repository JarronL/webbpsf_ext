# Import libraries
from copy import deepcopy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os, sys
import six

import webbpsf, poppy, pysiaf

try:
    from webbpsf.webbpsf_core import get_siaf_with_caching
except ImportError:
    # In case user doesn't have the latest version of webbpsf installed
    import functools
    @functools.lru_cache
    def get_siaf_with_caching(instrname):
        """ Parsing and loading the SIAF information is particularly time consuming,
        (can be >0.1 s per call, so multiple invokations can be a large overhead)
        Therefore avoid unnecessarily reloading it by caching results.
        This is a small speed optimization. """
        return pysiaf.Siaf(instrname)

siaf_nrc = get_siaf_with_caching('NIRCam')
siaf_nis = get_siaf_with_caching('NIRISS')
siaf_mir = get_siaf_with_caching('MIRI')
siaf_nrs = get_siaf_with_caching('NIRSpec')
siaf_fgs = get_siaf_with_caching('FGS')

from . import conf
from .logging_utils import setup_logging

import logging
_log = logging.getLogger('webbpsf_ext')

# from . import synphot_ext as S
from . import synphot_ext
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    synphot_ext.download_cdbs_data(verbose=True)

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

def pix_ang_size(ap=None, sr=True, pixscale=None):
    """Angular area of pixel from aperture object
    
    If `sr=True` then return in sterradians,
    otherwise return in units of arcsec^2.

    Parameters
    ==========
    ap : pysiaf.Aperture
        Aperture object
    sr : bool
        Return in steradians? Default True.
    pixscale : float, array-like, or None
        Pixel scale in arcsec/pixel. If None, then
        use `ap.XSciScale` and `ap.YSciScale` to
        determine pixel scale. If `pixscale` is
        array-like, then assume (xscale, yscale).
    """
    sr2asec2 = 42545170296.1522
    
    # X and Y scale in arcsec / pixel
    if pixscale is not None:
        if isinstance(pixscale, (np.ndarray, list, tuple)):
            xscale, yscale = pixscale
        else:
            xscale = yscale = pixscale
    else:
        if ap is None:
            raise ValueError("Must specify either `ap` or `pixscale`.")
        xscale = ap.XSciScale
        yscale = ap.YSciScale
    
    # Area in sq arcsec
    area_asec2 = xscale * yscale
    
    if sr:
        # Convert to sterradian
        return area_asec2 / sr2asec2
    else:
        return area_asec2
    
def load_plt_style(style='webbpsf_ext.wext_style'):
    """
    Load the matplotlib style for spaceKLIP plots.
    
    Load the style sheet in `sk_style.mplstyle`, which is a modified version of the
    style sheet from the `webbpsf_ext` package.
    """
    plt.style.use(style)
