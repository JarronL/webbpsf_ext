import numpy as np
import os

import poppy

from . import conf
from .logging_utils import setup_logging
from .webbpsf_ext_core import NIRCam_ext
from .bandpasses import nircam_filter
from .bandpasses import nircam_com_th, nircam_com_nd

# The following won't work on readthedocs compilation
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if not on_rtd:
    # Grab WebbPSF assumed pixel scales
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    nc_temp = NIRCam_ext()
    setup_logging(log_prev, verbose=False)

    pixscale_SW = nc_temp._pixelscale_short
    pixscale_LW = nc_temp._pixelscale_long
    del nc_temp

def det_to_sci(image, detid):
    """ Detector to science orientation
    
    Reorient image from detector coordinates to 'sci' coordinate system.
    This places +V3 up and +V2 to the LEFT. Detector pixel (0,0) is assumed 
    to be in the bottom left. Simply performs axes flips.
    
    Parameters
    ----------
    image : ndarray
        Input image or cube to transform.
    detid : int or str
        NIRCam detector/SCA ID, either 481-490, NRCA1-NRCB5, or A1-B5.
        ALONG and BLONG variations are also accepted.
    """
    
    from .utils import get_detname

    # Get detector name returns NRCA1, NRCB1, NRCA5, etc
    detname = get_detname(detid, use_long=False)
    if 'NRC' in detname:
        detname = detname[3:]

    xflip = ['A1','A3','A5','B2','B4']
    yflip = ['A2','A4','B1','B3','B5']

    # Handle multiple array of images
    ndim = len(image.shape)
    if ndim==2:
        # Convert to image cube
        ny, nx = image.shape
        image = image.reshape([1,ny,nx])
    
    for s in xflip:
        if detname in s:
            image = image[:,:,::-1] 
    for s in yflip:
        if detname in s:
            image = image[:,::-1,:] 
    
    # Convert back to 2D if input was 2D
    if ndim==2:
        image = image.reshape([ny,nx])

    return image
    
def sci_to_det(image, detid):
    """ Science to detector orientation
    
    Reorient image from 'sci' coordinates to detector coordinate system.
    Assumes +V3 up and +V2 to the LEFT. The result places the detector
    pixel (0,0) in the bottom left. Simply performs axes flips.

    Parameters
    ----------
    image : ndarray
        Input image or cube to transform.
    detid : int or str
        NIRCam detector/SCA ID, either 481-490, NRCA1-NRCB5, or A1-B5.
        ALONG and BLONG variations are also accepted.
    """
    
    # Flips occur along the same axis and manner as in det_to_sci()
    return det_to_sci(image, detid)

def coron_trans(name, module='A', pixelscale=None, npix=None, oversample=1, 
    nd_squares=True, shift_x=None, shift_y=None, filter=None):
    """
    Build a transmission image of a coronagraphic mask spanning
    the 20" coronagraphic FoV.

    oversample is used only if pixelscale is set to None.

    Returns the intensity transmission (square of the amplitude transmission). 
    """

    from webbpsf.optics import NIRCam_BandLimitedCoron

    shifts = {'shift_x': shift_x, 'shift_y': shift_y}

    bar_offset = None
    if name=='MASK210R':
        pixscale = pixscale_SW
        channel = 'short'
        filter = 'F210M' if filter is None else filter
    elif name=='MASK335R':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F335M' if filter is None else filter
    elif name=='MASK430R':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F430M' if filter is None else filter
    elif name=='MASKSWB':
        pixscale = pixscale_SW
        channel = 'short'
        filter = 'F210M' if filter is None else filter
        bar_offset = 0
    elif name=='MASKLWB':
        pixscale = pixscale_LW
        channel = 'long'
        filter = 'F430M' if filter is None else filter
        bar_offset = 0

    if pixelscale is None:
        pixelscale = pixscale / oversample
        if npix is None:
            npix = 320 if channel=='long' else 640
            npix = int(npix * oversample + 0.5)
    elif npix is None:
        # default to 20" if pixelscale is set but no npix
        npix = int(20 / pixelscale + 0.5)

    mask = NIRCam_BandLimitedCoron(name=name, module=module, bar_offset=bar_offset, auto_offset=None, 
                                   nd_squares=nd_squares, **shifts)

    # Create wavefront to pass through mask and obtain transmission image
    bandpass = nircam_filter(filter)
    wavelength = bandpass.avgwave() / 1e10
    wave = poppy.Wavefront(wavelength=wavelength, npix=npix, pixelscale=pixelscale)
    
    # Square the amplitude transmission to get intensity transmission
    im = mask.get_transmission(wave)**2    

    return im

def build_mask(module='A', pixscale=None, filter=None, nd_squares=True):
    """Create coronagraphic mask image

    Return a truncated image of the full coronagraphic mask layout
    for a given module. Assumes each mask is exactly 20" across.

    +V3 is up, and +V2 is to the left.
    """
    if module=='A':
        names = ['MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
    elif module=='B':
        names = ['MASKSWB', 'MASKLWB', 'MASK430R', 'MASK335R', 'MASK210R']

    if pixscale is None:
        pixscale=pixscale_LW

    npix = int(20 / pixscale + 0.5)
    allims = []
    for name in names:
        res = coron_trans(name, module=module, pixelscale=pixscale, npix=npix, nd_squares=nd_squares)
        allims.append(res)
    im_out = np.concatenate(allims, axis=1)

    # Multiply COM throughputs sampled at filter wavelength
    if filter is not None:
        bandpass = nircam_filter(filter)
        w_um = bandpass.avgwave() / 1e4
        com_th = nircam_com_th(wave_out=w_um)
        com_nd = 10**(-1*nircam_com_nd(wave_out=w_um))

        ind_nd = (im_out<0.0011) & (im_out>0.0009)
        im_out[ind_nd] = com_nd
        im_out *= com_th

    return im_out

def build_mask_detid(detid, oversample=1, ref_mask=None, pupil=None, filter=None, 
    nd_squares=True, mask_holder=True):
    """Create mask image for a given detector

    Return a full coronagraphic mask image as seen by a given SCA.
    +V3 is up, and +V2 is to the left ('sci' coordinates).

    Parameters
    ----------
    detid : str
        Name of detector, 'A1', A2', ... 'A5' (or 'ALONG'), etc.
    oversample : float
        How much to oversample output mask relative to detector sampling.
    ref_mask : str or None
        Reference mask for placement of coronagraphic mask elements.
        If None, then defaults are chosen for each detector.
    pupil : str or None
        Which Lyot pupil stop is being used? This affects holder placement.
        If None, then defaults based on ref_mask.
    """

    from .image_manip import pad_or_cut_to_size

    names = ['A1', 'A2', 'A3', 'A4', 'A5',
             'B1', 'B2', 'B3', 'B4', 'B5']

    # In case input is 'NRC??'
    if 'NRC' in detid:
        detid = detid[3:]

    # Convert ALONG to A5 name
    module = detid[0]
    detid = '{}5'.format(module) if 'LONG' in detid else detid

    # Make sure we have a valid name
    if detid not in names:
        raise ValueError("Invalid detid: {0} \n  Valid names are: {1}" \
              .format(detid, ', '.join(names)))

    pixscale = pixscale_LW if '5' in detid else pixscale_SW
    pixscale_over = pixscale / oversample

    # Build the full mask
    xpix = ypix = 2048
    xpix_over = int(xpix * oversample)
    ypix_over = int(ypix * oversample)

    cmask = np.ones([ypix_over, xpix_over], dtype='float64')

    # These detectors don't see any of the mask structure
    if detid in ['A1', 'A3', 'B2', 'B4']:
        return cmask

    if detid=='A2':
        cnames = ['MASK210R', 'MASK335R', 'MASK430R']
        ref_mask = 'MASK210R' if ref_mask is None else ref_mask
    elif detid=='A4':
        cnames = ['MASK430R', 'MASKSWB', 'MASKLWB']
        ref_mask = 'MASKSWB' if ref_mask is None else ref_mask
    elif detid=='A5':
        cnames = ['MASK210R', 'MASK335R', 'MASK430R', 'MASKSWB', 'MASKLWB']
        ref_mask = 'MASK430R' if ref_mask is None else ref_mask
    elif detid=='B1':
        cnames = ['MASK430R', 'MASK335R', 'MASK210R']
        ref_mask = 'MASK210R' if ref_mask is None else ref_mask
    elif detid=='B3':
        cnames = ['MASKSWB', 'MASKLWB', 'MASK430R']
        ref_mask = 'MASKSWB' if ref_mask is None else ref_mask
    elif detid=='B5':
        cnames = ['MASKSWB', 'MASKLWB', 'MASK430R', 'MASK335R', 'MASK210R']
        ref_mask = 'MASK430R' if ref_mask is None else ref_mask

    # Generate sub-images for each aperture
    # npix = int(ypix / len(cnames))
    npix = int(20.5 / pixscale_over + 0.5)
    npix_large = int(26 / pixscale_over + 0.5)
    allims = []
    for cname in cnames:
        res = coron_trans(cname, module=module, pixelscale=pixscale_over, npix=npix_large, 
                          filter=filter, nd_squares=nd_squares)
        allims.append(res)
    
    if pupil is None:
        pupil = 'WEDGELYOT' if ('WB' in ref_mask) else 'CIRCLYOT'

    # For each sub-image, expand and move to correct location
    channel = 'LW' if '5' in detid else 'SW'
    for i, name in enumerate(cnames):
        cdict = coron_ap_locs(module, channel, name, pupil=pupil, full=False)
        # Crop off large size
        im_crop = pad_or_cut_to_size(allims[i], (npix, npix_large))
        # Expand and offset
        xsci, ysci = cdict['cen_sci']
        xoff = xsci*oversample - ypix_over/2
        yoff = ysci*oversample - xpix_over/2
        im_expand = pad_or_cut_to_size(im_crop+1000, (ypix_over, xpix_over), offset_vals=(yoff,xoff))
        ind_good = ((cmask<100) & (im_expand>100)) | ((cmask==1001) & (im_expand>100))
        cmask[ind_good] = im_expand[ind_good]

    # Remove offsets
    cmask[cmask>100] = cmask[cmask>100] - 1000

    # Multiply COM throughputs sampled at filter wavelength
    if filter is not None:
        bandpass = nircam_filter(filter)
        w_um = bandpass.avgwave() / 1e4
        com_th = nircam_com_th(wave_out=w_um)
        com_nd = 10**(-1*nircam_com_nd(wave_out=w_um))

        ind_nd = (cmask<0.0011) & (cmask>0.0009)
        cmask[ind_nd] = com_nd
        cmask *= com_th

    # Place cmask in detector coords
    cmask = sci_to_det(cmask, detid)

    ############################################
    # Place blocked region from coronagraph holder
    # Also ensure region outside of COM has throughput=1
    if mask_holder:
        if detid=='A2':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(920*oversample), int(390*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:,0:i2]  = 1
                i1 = int(220*oversample)
                cmask[0:i1,:] = 0
                i2 = int(974*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(935*oversample), int(393*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:, 0:i2] = 1
                i1 = int(235*oversample)
                cmask[0:i1,:] = 0
                i2 = int(985*oversample)
                cmask[i2:,:] = 1
                
        elif detid=='A4':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(920*oversample), int(1463*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(220*oversample)
                cmask[0:i1,:] = 0
                i2 = int(974*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(935*oversample), int(1465*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(235*oversample)
                cmask[0:i1,:] = 0
                i2 = int(985*oversample)
                cmask[i2:,:] = 1
                
        elif detid=='A5':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(1480*oversample), int(270*oversample)]
                cmask[i1:,0:i2]  = 0
                cmask[0:i1,0:i2] = 1
                i1, i2 = [int(1480*oversample), int(1880*oversample)]
                cmask[i1:,i2:]  = 0
                cmask[0:i1,i2:] = 1
                i1 = int(1825*oversample)
                cmask[i1:,:] = 0
                i2 = int(1452*oversample)
                cmask[0:i2,:] = 1
            else:
                i1, i2 = [int(1485*oversample), int(275*oversample)]
                cmask[i1:,0:i2]  = 0
                cmask[0:i1,0:i2] = 1
                i1, i2 = [int(1485*oversample), int(1883*oversample)]
                cmask[i1:,i2:]  = 0
                cmask[0:i1,i2:] = 1
                i1 = int(1830*oversample)
                cmask[i1:,:] = 0
                i2 = int(1462*oversample)
                cmask[0:i2,:] = 1
                
        elif detid=='B1':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(910*oversample), int(1615*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:,i2:]  = 1
                i1 = int(210*oversample)
                cmask[0:i1,:] = 0
                i2 = int(956*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(905*oversample), int(1609*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:,i2:]  = 1
                i1 = int(205*oversample)
                cmask[0:i1,:] = 0
                i2 = int(951*oversample)
                cmask[i2:,:] = 1

        elif detid=='B3':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(920*oversample), int(551*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:,0:i2]  = 1
                i1 = int(210*oversample)
                cmask[0:i1,:] = 0
                i2 = int(966*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(920*oversample), int(548*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:,0:i2]  = 1
                i1 = int(210*oversample)
                cmask[0:i1,:] = 0
                i2 = int(963*oversample)
                cmask[i2:,:] = 1

        elif detid=='B5':
            if 'CIRCLYOT' in pupil:
                i1, i2 = [int(555*oversample), int(207*oversample)]
                cmask[0:i1,0:i2] = 0
                cmask[i1:, 0:i2] = 1
                i1, i2 = [int(545*oversample), int(1815*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(215*oversample)
                cmask[0:i1,:] = 0
                i2 = int(578*oversample)
                cmask[i2:,:] = 1
            else:
                i1, i2 = [int(555*oversample), int(211*oversample)]
                cmask[0:i1,0:i2] = 0 
                cmask[i1:, 0:i2] = 1
                i1, i2 = [int(545*oversample), int(1819*oversample)]
                cmask[0:i1,i2:] = 0
                cmask[i1:, i2:] = 1
                i1 = int(215*oversample)
                cmask[0:i1,:] = 0
                i2 = int(578*oversample)
                cmask[i2:,:] = 1

    ############################################
    # Fix SW/LW wedge abuttment
    if detid=='A4':
        if 'CIRCLYOT' in pupil:
            x0 = 819
            x1 = 809
            x2 = x1 + 10
        else:
            x0 = 821
            x1 = 812
            x2 = x1 + 9
        y1, y2 = (400, 650)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])
    elif detid=='A5':
        if 'CIRCLYOT' in pupil:
            x0 = 587
            x1 = x0 + 1
            x2 = x1 + 5
        else:
            x0 = 592
            x1 = x0 + 1
            x2 = x1 + 5
        y1, y2 = (1600, 1750)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])
            
    elif detid=='B3':
        if 'CIRCLYOT' in pupil:
            x0 = 1210
            x1 = 1196
            x2 = x1 + 14
        else:
            x0 = 1204
            x1 = 1192
            x2 = x1 + 12
        y1, y2 = (350, 650)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])
    elif detid=='B5':
        if 'CIRCLYOT' in pupil:
            x0 = 531
            x1 = 525
            x2 = x1 + 6
        else:
            x0 = 535
            x1 = 529
            x2 = x1 + 6
        y1, y2 = (300, 420)
        ix0 = int(x0*oversample)
        iy1, iy2 = int(y1*oversample), int(y2*oversample)
        ix1, ix2 = int(x1*oversample), int(x2*oversample)
        cmask[iy1:iy2,ix1:ix2] = cmask[iy1:iy2,ix0].reshape([-1,1])

    # Convert back to 'sci' orientation
    cmask = det_to_sci(cmask, detid)

    return cmask


def coron_ap_locs(module, channel, mask, pupil=None, full=False):
    """Coronagraph mask aperture locations and sizes

    Returns a dictionary of the detector aperture sizes
    and locations. Attributes 'cen' and 'loc' are in terms
    of (x,y) detector pixels. 'cen_sci' is sci coords location.
    """

    if channel=='long':
        channel = 'LW'
    elif channel=='short':
        channel = 'SW'
    
    if pupil is None:
        pupil = 'WEDGELYOT' if 'WB' in mask else 'CIRCLYOT'

    if module=='A':
        if channel=='SW':
            if '210R' in mask:
                cdict_rnd = {'det':'A2', 'cen':(712,525), 'size':640}
                cdict_bar = {'det':'A2', 'cen':(716,536), 'size':640}
            elif '335R' in mask:
                cdict_rnd = {'det':'A2', 'cen':(1368,525), 'size':640}
                cdict_bar = {'det':'A2', 'cen':(1372,536), 'size':640}
            elif '430R' in mask:
                cdict_rnd = {'det':'A2', 'cen':(2025,525), 'size':640}
                cdict_bar = {'det':'A2', 'cen':(2029,536), 'size':640}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'A4', 'cen':(487,523), 'size':640}
                cdict_bar = {'det':'A4', 'cen':(490,536), 'size':640}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'A4', 'cen':(1141,523), 'size':640}
                cdict_bar = {'det':'A4', 'cen':(1143,536), 'size':640}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        elif channel=='LW':
            if '210R' in mask:
                cdict_rnd = {'det':'A5', 'cen':(1720, 1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(1725, 1682), 'size':320}
            elif '335R' in mask:
                cdict_rnd = {'det':'A5', 'cen':(1397,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(1402,1682), 'size':320}
            elif '430R' in mask:
                cdict_rnd = {'det':'A5', 'cen':(1074,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(1078,1682), 'size':320}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'A5', 'cen':(752,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(757,1682), 'size':320}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'A5', 'cen':(430,1672), 'size':320}
                cdict_bar = {'det':'A5', 'cen':(435,1682), 'size':320}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        else:
            raise ValueError('Channel {} not recognized'.format(channel))


    elif module=='B':
        if channel=='SW':
            if '210R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(1293,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(1287,508), 'size':640}
            elif '335R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(637,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(632,508), 'size':640}
            elif '430R' in mask:
                cdict_rnd = {'det':'B1', 'cen':(-20,513), 'size':640}
                cdict_bar = {'det':'B1', 'cen':(-25,508), 'size':640}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'B3', 'cen':(874,519), 'size':640}
                cdict_bar = {'det':'B3', 'cen':(870,516), 'size':640}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'B3', 'cen':(1532,519), 'size':640}
                cdict_bar = {'det':'B3', 'cen':(1526,516), 'size':640}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        elif channel=='LW':
            if '210R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1656,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1660,360), 'size':320}
            elif '335R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1334,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1338,360), 'size':320}
            elif '430R' in mask:
                cdict_rnd = {'det':'B5', 'cen':(1012,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(1015,360), 'size':320}
            elif 'SWB' in mask:
                cdict_rnd = {'det':'B5', 'cen':(366,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(370,360), 'size':320}
            elif 'LWB' in mask:
                cdict_rnd = {'det':'B5', 'cen':(689,360), 'size':320}
                cdict_bar = {'det':'B5', 'cen':(693,360), 'size':320}
            else:
                raise ValueError('Mask {} not recognized for {} channel'\
                                 .format(mask, channel))
        else:
            raise ValueError('Channel {} not recognized'.format(channel))

    else:
        raise ValueError('Module {} not recognized'.format(module))

    # Choose whether to use round or bar Lyot mask
    cdict = cdict_rnd if 'CIRC' in pupil else cdict_bar

    x0, y0 = np.array(cdict['cen']) - cdict['size']/2
    cdict['loc'] = (int(x0), int(y0))


    # Add in 'sci' coordinates (V2/V3 orientation)
    # X is flipped for A5, Y is flipped for all others
    cen = cdict['cen']
    if cdict['det'] == 'A5':
        cdict['cen_sci'] = (2048-cen[0], cen[1])
    else:
        cdict['cen_sci'] = (cen[0], 2048-cen[1])

    if full:
        cdict['size'] = 2048
        cdict['loc'] = (0,0)

    return cdict

def coron_detector(mask, module, channel=None):
    """
    Return detector name for a given coronagraphic mask, module,
    and channel.
    """
    
    # Grab default channel
    if channel is None:
        if ('210R' in mask) or ('SW' in mask):
            channel = 'SW'
        else:
            channel = 'LW'
    
    # If LW, always A5 or B5
    # If SW, bar masks are A4/B3, round masks A2/B1; M430R is invalid
    if channel=='LW':
        detname = module + '5'
    # elif (channel=='SW') and ('430R' in mask):
    #     raise AttributeError("MASK430R not valid for SW channel")
    else:
        if module=='A':
            detname = 'A2' if mask[-1]=='R' else 'A4'
        else:
            detname = 'B1' if mask[-1]=='R' else 'B3'
            
    return detname

def gen_coron_mask(apname, image_mask=None, pupil_mask=None, filter=None,
                   oversample=1, **kwargs):
    """
    Generate coronagraphic mask transmission images.

    Output images are in 'sci' coordinates.
    """

    from .utils import siaf_nrc
    from .image_manip import frebin

    if image_mask is None:
        if '335R' in apname:
            mask = 'MASK335R'
        elif '430R' in apname:
            mask = 'MASK430R'
        elif 'SWB' in apname:
            mask = 'MASKSWB'
        elif 'LWB' in apname:
            mask = 'MASKLWB'
        elif '210R' in apname:
            mask = 'MASK210R'
        else:
            raise ValueError("Coron mask can not be determined from {}".format(apname))
    else:
        mask = image_mask
    
    if pupil_mask is None:
        pupil = 'WEDGELYOT' if ('WB' in mask) else 'CIRCLYOT'

    ap = siaf_nrc[apname]
    detid = apname[0:5]

    x0, y0 = np.array(ap.dms_corner()) - 1
    xpix, ypix = (ap.XSciSize, ap.YSciSize)

    # im_det  = build_mask_detid(detid, oversample=1, pupil=pupil)
    im_over = build_mask_detid(detid, ref_mask=mask, oversample=oversample, 
                                pupil=pupil, filter=filter)
    # Convert to det coords and crop
    # im_det  = sci_to_det(im_det, detid)
    im_over = sci_to_det(im_over, detid)
    im_det = frebin(im_over, scale=1/oversample, total=False)

    return im_det[y0:y0+ypix, x0:x0+xpix]

def gen_coron_mask_ndonly(apname, **kwargs):
    
    from .utils import siaf_nrc
    from .coords import dist_image, rtheta_to_xy

    cmask = gen_coron_mask(apname, **kwargs)

    ap = siaf_nrc[apname]
    oversample = kwargs.get('oversample', 1)
    pixscale_over = ap.YSciScale / oversample

    center = np.array([ap.XSciRef, ap.YSciRef]) * oversample
    r, th = dist_image(cmask, pixscale=pixscale_over, center=center, return_theta=True)
    x_asec, y_asec = rtheta_to_xy(r, th)

    # Mask out everything except the ND squares and mask holder
    ind = (np.abs(y_asec)<4.5) & (cmask>0)

    # Get transmission value
    filter = kwargs.get('filter', None)
    if filter is None:
        com_th = 1
    else:
        bandpass = nircam_filter(filter)
        w_um = bandpass.avgwave() / 1e4
        com_th = nircam_com_th(wave_out=w_um)

    cmask[ind] = com_th

    return cmask