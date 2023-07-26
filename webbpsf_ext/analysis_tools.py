import numpy as np
import os
from tqdm.auto import tqdm

from . import robust

from astropy.io import fits
from skimage.registration import phase_cross_correlation

# Create NRC SIAF class
from .utils import get_one_siaf
nrc_siaf = get_one_siaf(instrument='NIRCam')

import logging
# Define logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)


def get_radial_profiles(im, center=None, binsize=1, bpmask=None, 
                        radin=0, radout=None, use_poppy=False):
    """Get radial profiles (average flux, EE, std)
    
    Take the sum of pixels within increasing radius.

    Pass bpmask to set certain pixels to 0.
    """

    from webbpsf_ext.maths import binned_statistic, dist_image

    # Exclude pixels with NaNs
    bpmask_nans = np.isnan(im)
    if bpmask is None:
        bpmask = bpmask_nans
    else:
        bpmask = bpmask | bpmask_nans

    im = im.copy()
    im[bpmask] = np.nan

    # Set values interior to radin equal to 0
    if radin>0:
        rdist = dist_image(im)
        im[rdist<radin]=0

    if use_poppy:
        from poppy.utils import radial_profile
        hdu = fits.PrimaryHDU(im)
        hdu.header['PIXELSCL'] = 1
        hdul = fits.HDUList([hdu])
        res = radial_profile(hdul, ee=True, center=center, binsize=binsize)
        _, std = radial_profile(hdul, ee=False, stddev=True, center=center, binsize=binsize)
        hdul.close()

        rr, rp, ee = res

        # Crop outer radii
        radout = rr.max() if radout is None else radout
        ind_use = rr <= radout

        rr = rr[ind_use]
        ee = ee[ind_use]
        rp = rp[ind_use]
        std = std[ind_use]
    else:
        # Get distances for each pixel
        rho = dist_image(im, center=center)
        # Break into radial bins
        radial_bins = np.arange(rho.min(), rho.max() + binsize, binsize)
        # Sum pixels within each bin
        radial_sum_im = binned_statistic(rho, im, func=np.nansum, bins=radial_bins)
        # Cumulative sum to get encircled energy
        ee_im = np.cumsum(radial_sum_im)

        # Standard deviation of pixels within each bin
        std = binned_statistic(rho, im, func=robust.medabsdev, bins=radial_bins)
        
        radial_bins_mid = radial_bins[:-1] + binsize / 2

        radout = radial_bins_mid.max() if radout is None else radout
        ind_use = radial_bins_mid <= radout

        rr = radial_bins_mid[ind_use]
        ee = ee_im[ind_use]
        rp = radial_sum_im[ind_use]
        std = std[ind_use]

    return rr, ee, rp, std

def get_encircled_energy(im, center=None, binsize=1, 
    return_radial_profile=False, bpmask=None, 
    radin=0, radout=None, use_poppy=False):
    """Get encircled energy and optional radial profiles
    
    Take the sum of pixels within increasing radius.

    Pass bpmask to set certain pixels to 0.
    """

    res = get_radial_profiles(im, center=center, binsize=binsize, 
                              bpmask=bpmask, radin=radin, radout=radout, 
                              use_poppy=use_poppy)
    
    rr, ee, rp, _ = res

    if return_radial_profile:
        return rr, ee, rp
    else:
        return rr, ee


def ipc_info(sca_input):
    """ Return IPC coefficients and kernel for a given SCA

    Returns IPC coefficients (a1, a2) and IPC kernel for a given SCA.
    Default IPC coefficients are (0.005, 0.0003) if SCA is not found.
    The IPC kernel is a 3x3 array with the center pixel being 1-4*(a1+a2):

        [[a2, a1, a2],
         [a1, a0, a1],
         [a2, a1, a2]]

    where a0 = 1-4*(a1+a2).
    
    Parameters
    ----------
    sca_input : str
        Name of NIRCam SCA. Can be in any format such as A5, NRCA5, NRCALONG.

    Returns
    -------
    ipc : tuple
        Tuple of IPC coefficients (a1, a2)
    kipc : ndarray
        3x3 IPC kernel
    """

    from .utils import get_detname

    # Returns NRC[A-B][1-4,LONG]
    sca = get_detname(sca_input, use_long=False).lower()

    ipc_dict = {
        'nrca1': (0.00488, 0.00027), 'nrca2': (0.00516, 0.00031),
        'nrca3': (0.00568, 0.00033), 'nrca4': (0.00557, 0.00039),
        'nrcb1': (0.00511, 0.00028), 'nrcb2': (0.00464, 0.00025),
        'nrcb3': (0.00542, 0.00033), 'nrcb4': (0.00570, 0.00032),
        'nrca5': (0.00600, 0.00039), 'nrcb5': (0.00554, 0.00037),
        }

    keys = list(ipc_dict.keys())
    if sca not in keys:
        a1, a2 = (0.005,0.0003)
        _log.warn(f"{sca_input} ({sca}) does not match known NIRCam SCA. \
                    Defaulting to ({a1:.4f}, {a2:.4f}).")
    else:
        a1, a2 = ipc_dict.get(sca)

    # Create IPC kernel
    kipc = np.array([[a2,a1,a2], [a1,1-4*(a1+a2),a1], [a2,a1,a2]])

    return (a1, a2), kipc

def ppc_info(sca_input):
    """ Return PPC coefficients and kernel for a given SCA

    Returns PPC coefficients (ppc_frac) and PPC kernel for a given SCA.
    Defaults to 0.001 if SCA is not found.

    PPC is dependent on readout direction, with some fraction of the signal
    being transferred to the trailing pixel.

    Parameters
    ----------
    sca_input : str
        Name of NIRCam SCA. Can be in any format such as A5, NRCA5, NRCALONG.

    Returns
    -------
    ppc_frac : float

    """

    from .utils import get_detname

    # Returns NRC[A-B][1-4,LONG]
    sca = get_detname(sca_input, use_long=False).lower()

    ppc_dict = {
        'nrca1': 0.00065, 'nrca2': 0.00069,
        'nrca3': 0.00023, 'nrca4': 0.00069,
        'nrcb1': 0.00033, 'nrcb2': 0.00063,
        'nrcb3': 0.00034, 'nrcb4': 0.00078,
        'nrca5': 0.00123, 'nrcb5': 0.00140,
        }
    
    keys = list(ppc_dict.keys())
    if sca not in keys:
        ppc_frac = 0.001
        _log.warn(f"{sca_input} ({sca}) does not match known NIRCam SCA. \
                    Defaulting to {ppc_frac:.3f}.")
    else:
        ppc_frac = ppc_dict.get(sca)

    kppc = np.array([[0,0,0], [0,1-ppc_frac,ppc_frac], [0,0,0]])

    return ppc_frac, kppc


def nrc_ref_info(apname, orientation='sci'):
    """Get reference pixel information for a given aperture

    Returns number of reference pixels around subarray border
    [lower, upper, left, right] in either 'sci' or 'det' orientation.
    Default is 'sci' orientation.

    Parameters
    ----------
    apname : str
        Name of NIRCam SIAF aperture.
    orientation : str
        Orientation of subarray. Either 'sci' or 'det'.
    """

    ap = nrc_siaf[apname]

    det_size = 2048
    xpix = ap.XSciSize
    ypix = ap.YSciSize

    xcorn, ycorn = ap.corners('det')

    x1 = int(np.min(xcorn) - 0.5)
    y1 = int(np.min(ycorn) - 0.5)

    x2 = x1 + xpix
    y2 = y1 + ypix

    w = 4 # Width of ref pixel border
    lower = int(w-y1)
    upper = int(w-(det_size-y2))
    left  = int(w-x1)
    right = int(w-(det_size-x2))
    # Keep as list rather than np.array to prevent type convesion to int64
    ref_all = [lower,upper,left,right]
    for i, r in enumerate(ref_all):
        if r<0:
            ref_all[i] = 0

    # Flip for orientation depending on detector
    if orientation=='sci':
        det = ap.AperName[3:5]
        xflip = ['A1','A3','A5','B2','B4']
        yflip = ['A2','A4','B1','B3','B5']

        if det in xflip:
            # Flip left/right
            ref_all[2:] = ref_all[2:][::-1]
        elif det in yflip:
            # Flip top/bottom
            ref_all[:2] = ref_all[:2][::-1]

    return ref_all
