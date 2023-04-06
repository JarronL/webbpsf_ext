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

    from .utils import get_detname

    # Returns NRC[A-B][1-4,LONG]
    sca = get_detname(sca_input, use_long=False).lower()

    ipc_dict = {
        'nrca1': (0.0049, 0.0003), 'nrca2': (0.0052, 0.0003),
        'nrca3': (0.0057, 0.0003), 'nrca4': (0.0056, 0.0004),
        'nrcb1': (0.0051, 0.0003), 'nrcb2': (0.0046, 0.0003),
        'nrcb3': (0.0054, 0.0003), 'nrcb4': (0.0057, 0.0003),
        'nrca5': (0.0060, 0.0004), 'nrcb5': (0.0055, 0.0004),
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
