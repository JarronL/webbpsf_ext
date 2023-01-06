import numpy as np
import os
from tqdm.auto import tqdm

from .image_manip import fourier_imshift, fshift
from .image_manip import frebin, pad_or_cut_to_size, bp_fix
from .coords import dist_image
from .maths import round_int

from astropy.io import fits
from skimage.registration import phase_cross_correlation

# Create NRC SIAF class
from .utils import get_one_siaf
nrc_siaf = get_one_siaf(instrument='NIRCam')

import logging
# Define logging
_log = logging.getLogger(__name__)
_log.setLevel(logging.INFO)

###########################################################################
#    Target Acquisition
###########################################################################

def get_ictm_event_log(startdate, enddate, hdr=None, mast_api_token=None, verbose=False):
    """"""

    from datetime import datetime, timedelta, timezone
    from requests import Session

    # parameters
    mnemonic = 'ICTM_EVENT_MSG'

    # constants
    base = 'https://mast.stsci.edu/jwst/api/v0.1/Download/file?uri=mast:jwstedb'
    mastfmt = '%Y%m%dT%H%M%S'
    tz_utc = timezone(timedelta(hours=0))

    # establish MAST session
    session = Session()

    # Attempt to find MAST token if set to None
    if mast_api_token is None:
        mast_api_token = os.environ.get('MAST_API_TOKEN')
        # NOTE: MAST token is no longer strictly necessary (I think?)
        # if mast_api_token is None:
        #     raise ValueError("Must define MAST_API_TOKEN env variable or specify mast_api_token parameter")

    # Update token
    if mast_api_token is not None:
        session.headers.update({'Authorization': f'token {mast_api_token}'})

    # Determine date range to grab data
    if hdr is not None:
        startdate = hdr['VSTSTART']
        enddate = hdr['VISITEND']
    
    startdate = startdate.replace(' ', '+')
    idot = startdate.index('.')
    startdate = startdate[0:idot]
    enddate = enddate.replace(' ', '+')
    idot = enddate.index('.')
    enddate = enddate[0:idot]

    # fetch event messages from MAST engineering database (lags FOS EDB)
    start = datetime.fromisoformat(startdate)
    end = datetime.now(tz=tz_utc) if enddate is None else datetime.fromisoformat(enddate)
    startstr = start.strftime(mastfmt)
    endstr = end.strftime(mastfmt)
    filename = f'{mnemonic}-{startstr}-{endstr}.csv'
    url = f'{base}/{filename}'

    if verbose:
        _log.info(f"Retrieving {url}")
    response = session.get(url)
    if response.status_code == 401:
        exit('HTTPError 401 - Check your MAST token and EDB authorization.')
    response.raise_for_status()
    lines = response.content.decode('utf-8').splitlines()

    return lines


def find_centroid_det(eventlog, selected_visit_id):
    """Get centroid position of TA as reported in JWST event logs"""

    from csv import reader
    from datetime import datetime

    # parse response (ignoring header line) and print new event messages
    vid = ''
    in_selected_visit = False
    ta_only = True
    in_ta = False

    for value in reader(eventlog, delimiter=',', quotechar='"'):
        if in_selected_visit and ((not ta_only) or in_ta) :
            # print(value[0][0:22], "\t", value[2])
            
            # Print coordinate location info
            val_str = value[2]
            if ('postage-stamp coord' in val_str) or ('detector coord' in val_str): 
                _log.info(val_str)
        
            # Parse centroid position reported in detector coordinates
            if 'detector coord (colCentroid, rowCentroid)' in val_str:
                val_str_list = val_str.split('=')
                xcen, ycen = val_str_list[1].split(',')
                ind1 = xcen.find('(')
                xcen = xcen[ind1+1:]
                ind2 = ycen.find(')')
                ycen = ycen[0:ind2]

                return float(xcen), float(ycen)

        if value[2][:6] == 'VISIT ':
            if value[2][-7:] == 'STARTED':
                vstart = 'T'.join(value[0].split())[:-3]
                vid = value[2].split()[1]

                if vid==selected_visit_id:
                    _log.debug(f"VISIT {selected_visit_id} START FOUND at {vstart}")
                    in_selected_visit = True
                    # if ta_only:
                    #     print("Only displaying TARGET ACQUISITION RESULTS:")

            elif value[2][-5:] == 'ENDED' and in_selected_visit:
                assert vid == value[2].split()[1]
                assert selected_visit_id  == value[2].split()[1]

                vend = 'T'.join(value[0].split())[:-3]
                _log.debug(f"VISIT {selected_visit_id} END FOUND at {vend}")

                in_selected_visit = False
        elif value[2][:31] == f'Script terminated: {vid}':
            if value[2][-5:] == 'ERROR':
                script = value[2].split(':')[2]
                vend = 'T'.join(value[0].split())[:-3]
                dur = datetime.fromisoformat(vend) - datetime.fromisoformat(vstart)
                note = f'Halt in {script}'
                in_selected_visit = False
        elif in_selected_visit and value[2].startswith('*'): 
            # this string is used to mark the start and end of TA sections
            in_ta = not in_ta

def diff_ta_data(uncal_data):
    """Onboard algorithm to difference TA data"""
    
    data = uncal_data.astype('float')
    
    nint, ng, ny, nx = data.shape
    im1 = data[0,-1]    - data[0,ng//2]
    im2 = data[0,ng//2] - data[0,0]
    
    return np.minimum(im1,im2)

def read_ta_conf_files(indir, pid, obsid, sca, bpfix=False):
    """Store all TA and Conf data into a dictionary"""

    sca = sca.lower()
    allfiles_uncals = np.sort([f for f in os.listdir(indir) if f'{sca}_uncal.fits' in f])
    allfiles_rates = np.sort([f for f in os.listdir(indir) if f'{sca}_rate.fits' in f])

    # Select only Obs ID files
    file_start = f'jw{pid:05d}{obsid:03d}'
    obsfiles_uncals = np.sort([f for f in allfiles_uncals if file_start in f])
    obsfiles_rates = np.sort([f for f in allfiles_rates if file_start in f])

    # For TA files, search for EXP_TYPE=*_TACQ
    fta = None
    for f in obsfiles_uncals:
        fpath = os.path.join(indir, f)
        hdr = fits.getheader(fpath, ext=0)
        if '_TACQ' in hdr['EXP_TYPE']:
            fta = f
            break
    if fta is None:
        print("WARNING - TA uncal.fits not found, attempting _rate.fits...")
        for f in obsfiles_rates:
            fpath = os.path.join(indir, f)
            hdr = fits.getheader(fpath, ext=0)
            if '_TACQ' in hdr['EXP_TYPE']:
                fta = f
                break

    # Find Conf1 and Conf2 in rate.fits files
    for f in obsfiles_rates:
        fpath = os.path.join(indir, f)
        hdr = fits.getheader(fpath, ext=0)
        # 1st TACONFIM has TAMASK in APERNAME while 2nd is a science apname
        if ('_TACONFIRM' in hdr['EXP_TYPE']) and ('TAMASK' in hdr['APERNAME']):
            fconf1 = f
        elif ('_TACONFIRM' in hdr['EXP_TYPE']):
            fconf2 = f

    ta_dict = {
        'dta':    {'file': fta, 'type': 'Target Acq'},
        'dconf1': {'file': fconf1, 'type': 'TA Conf1'},
        'dconf2': {'file': fconf2, 'type': 'TA Conf2'},
    }
    
    # Build dictionary of data and header info
    for k in ta_dict.keys():
        d = ta_dict[k]

        f = d['file']
        hdul = fits.open(indir + f)
        # Get data and take diff if uncal
        data = hdul[1].data.astype('float')        
        if 'uncal.fits' in f:
            data = diff_ta_data(data)
        
        d['data'] = data
        d['hdr0'] = hdul[0].header
        d['hdr1'] = hdul[1].header
        hdul.close()

        d['apname'] = d['hdr0']['APERNAME']
        d['ap'] = nrc_siaf[d['apname']]

        # bad pixel fixing for TA confirmation around a 100=pixel region
        if bpfix and ('conf' in k):
            im = crop_observation(d['data'], d['ap'], 100)
            # Perform pixel fixing in place
            _ = bp_fix(im, sigclip=10, niter=10, in_place=True)
        
    return ta_dict


def read_sgd_files(indir, pid, obsid, sca, expid=None, filter=None, fext='_rate.fits'):
    """Store all observations of a given filter into a dictionary
    
    Mostly used for grabbing a series of SGD data to analyze.

    Parameters
    ==========
    indir : str
        Input directory
    pid : int
        Program ID number
    obsid : int
        Observation number
    sca : str
        SCA name, such as a1, a2, a3, a4, along, etc
    expid : str or None
        Five-digit string of exposure ID: {VisitGrp:02d}{SeqID:01d}{ActID:02d}.
        Second sub-string in filesname, usually 03105 or 03106.
        Should be set to None if filter is specified.
    filter : str or None
        Name of filter element.
    fext : str
        File extension, such as _uncal.fits, _rate.fits, _cal.fits, etc.
    """

    sca = sca.lower()
    file_end = f'{sca}_{fext}'

    allfiles_rates = np.sort([f for f in os.listdir(indir) if file_end in f])
    
    # Select only SGD files
    file_start = f'jw{pid:05d}{obsid:03d}'
    obsfiles_rates = np.sort([f for f in allfiles_rates if (file_start in f)])

    if filter is None and expid is None:
        print('WARNING: No filter or expid specificed. Will read in all SGD files.')
    elif (filter is not None) and (expid is not None):
        print('WARNING: Both filter and expid are specifed. If in conflict, expid takes priority.')

    if expid is not None:
        # Priority goes to expid value, ignore filters, and NRC_CORON header
        fsgd = np.sort([f for f in obsfiles_rates if (f'_{expid}_' in f)])
    else:
        fsgd = []
        for f in obsfiles_rates:
            fpath = os.path.join(indir, f)
            hdr = fits.getheader(fpath, ext=0)

            # Append if EXP_TYPE = NRC_CORON and specified filter matches
            # Will ta
            if ('_CORON' in hdr['EXP_TYPE']) and ((filter is None) or (hdr['FILTER']==filter)):
                fsgd.append(f)
        fsgd = np.sort(fsgd)
    
    sgd_dict = {}
    for i, f in enumerate(fsgd):
        fpath = os.path.join(indir, f)
        d = {'file': fpath}
        
        hdul = fits.open(fpath)
        d['data'] = hdul[1].data.astype('float')
        d['hdr0'] = hdul[0].header
        d['hdr1'] = hdul[1].header
        hdul.close()        
 
        d['apname'] = d['hdr0']['APERNAME']
        d['ap'] = nrc_siaf[d['apname']]
        
        sgd_dict[i] = d
        
    return sgd_dict

###########################################################################
#    Cross Correlations
###########################################################################

def correl_images(im1, im2, mask=None):
    """ Image correlation coefficient
    
    Calculate the 2D cross-correlation coefficient between two
    images or array of images. Images must have the same x and
    y dimensions and should alredy be aligned.
    
    Parameters
    ----------
    im1 : ndarray
        Single image or image cube (nz1, ny1, nx1).
    im2 : ndarray
        Single image or image cube (nz2, ny1, nx1). 
        If both im1 and im2 are cubes, then returns
        a matrix of  coefficients.
    mask : ndarry or None
        If set, then a binary mask of 1=True and 0=False.
        Excludes pixels marked with 0s/False. Must be same
        size/shape as images (ny1, nx1).
    """
    
    sh1 = im1.shape
    sh2 = im2.shape

    if len(sh1)==2:
        ny1, nx1 = sh1
        nz1 = 1
        im1.reshape([nz1,ny1,nx1])
    else:
        nz1, ny1, nx1 = sh1

    if len(sh2)==2:
        ny2, nx2 = sh2
        nz2 = 1
        im2.reshape([nz2,ny2,nx2])
    else:
        nz2, ny2, nx2 = sh2

    assert (nx1==nx2) and (ny1==ny2), "Input images must have same sizes"

    im1 = im1.reshape([nz1,-1])
    im2 = im2.reshape([nz2,-1])

    # Apply masking
    if mask is not None:
        im1 = im1[:, mask.ravel()]
        im2 = im2[:, mask.ravel()]

    # Subtract mean from each axes
    im1 = im1 - np.mean(im1, axis=1).reshape([-1,1])
    im2 = im2 - np.mean(im2, axis=1).reshape([-1,1])

    # Calculate numerators for each image pair
    correl_top = np.dot(im1, im2.T)

    # Calculate denominators for each image pair
    im1_tot = np.sum(im1**2, axis=1)
    im2_tot = np.sum(im2**2, axis=1)
    correl_bot = np.sqrt(np.multiply.outer(im1_tot, im2_tot))

    correl_fin = correl_top / correl_bot
    if correl_fin.size==1:
        return correl_fin.flatten()[0]
    else:
        return correl_fin.squeeze()

#### Best fit offset

def sample_crosscorr(corr, xcoarse, ycoarse, xfine, yfine):
    """Perform a cubic interpolation over the coarse grid"""
    
    from scipy.interpolate import griddata
    
    xycoarse = np.asarray(np.meshgrid(xcoarse, ycoarse)).reshape([2,-1]).transpose()

    # Sub-sampling shifts to interpolate over
    xv, yv = np.meshgrid(xfine, yfine)
    
    # Perform cubic interpolation
    corr_fine = griddata(xycoarse, corr.flatten(), (xv, yv), method='cubic')
    
    return corr_fine

def find_max_crosscorr(corr, xsh_arr, ysh_arr, sub_sample):
    """Interpolate finer grid onto cross corr map and location max position"""
    
    # Sub-sampling shifts to interpolate over
    # sub_sample = 0.01
    xsh_fine_vals = np.arange(xsh_arr[0],xsh_arr[-1],sub_sample)
    ysh_fine_vals = np.arange(ysh_arr[0],ysh_arr[-1],sub_sample)
    corr_all_fine = sample_crosscorr(corr,  xsh_arr, ysh_arr, xsh_fine_vals, ysh_fine_vals)

    # Fine position
    iymax, ixmax = np.argwhere(corr_all_fine==np.max(corr_all_fine))[0]
    xsh_fine, ysh_fine = xsh_fine_vals[ixmax], ysh_fine_vals[iymax]
    
    return xsh_fine, ysh_fine


def gen_psf_offsets(psf, crop=65, xlim_pix=(-3,3), ylim_pix=(-3,3), dxy=0.05,
    psf_osamp=1, shift_func=fourier_imshift, ipc_vals=None, kipc=None,
    prog_leave=False, **kwargs):
    """ Generate a series of downsampled cropped and shifted PSF images

    If fov_pix is odd, then crop should be odd. 
    If fov_pix is even, then crop should be even.
    
    Add IPC:
        Either ipc_vals = 0.006 or ipc_vals=[0.006,0.0004].
        The former add 0.6% to each side pixel, while the latter
        adds 0.04% to the corners.
    """
    
    from pynrc.simul.ngNRC import add_ipc

    psf_is_even = np.mod(psf.shape[0] / psf_osamp, 2) == 0
    psf_is_odd = not psf_is_even
    crop_is_even = np.mod(crop, 2) == 0
    crop_is_odd = not crop_is_even

    if (psf_is_even and crop_is_odd) or (psf_is_odd and crop_is_even):
        crop = crop + 1
        crop_is_even = np.mod(crop, 2) == 0
        crop_is_odd = not crop_is_even
        _log.warning('PSF and crop must both be even or odd. Incrementing crop by 1.')

    # Range of offsets to probe in fractional pixel steps
    xmin_pix, xmax_pix = xlim_pix
    ymin_pix, ymax_pix = ylim_pix

    # Pixel offsets
    xoff_pix = np.arange(xmin_pix, xmax_pix+dxy, dxy)
    yoff_pix = np.arange(ymin_pix, ymax_pix+dxy, dxy)

    # Create a grid and flatten
    xoff_all, yoff_all = np.meshgrid(xoff_pix, yoff_pix)
    xoff_all = xoff_all.flatten()
    yoff_all = yoff_all.flatten()
    
    # Make initial crop so we don't shift entire image
    crop_init = crop + int(2*(np.max(np.abs(np.concatenate([xoff_pix, yoff_pix]))) + 1))
    crop_init_over = crop_init * psf_osamp
    psf0 = crop_image(psf, crop_init_over)
    # psf0 = pad_or_cut_to_size(psf, crop_init_over)

    # Create a series of shifted PSFs to compare to images
    psf_sh_all = []
    for xoff, yoff in tqdm(zip(xoff_all, yoff_all), total=len(xoff_all), leave=prog_leave):
        xoff_over = xoff*psf_osamp
        yoff_over = yoff*psf_osamp
        crop_over = crop*psf_osamp

        psf_sh = crop_image(psf0, crop_over, xyloc=None, delx=-xoff_over, dely=-yoff_over,
                            shift_func=shift_func, pad=False, **kwargs)
        # psf_sh = pad_or_cut_to_size(psf0, crop_over, offset_vals=(-yoff_over,-xoff_over), 
        #                             shift_func=shift_func, pad=True)

        # Rebin to detector pixels
        psf_sh = frebin(psf_sh, scale=1/psf_osamp)
        psf_sh_all.append(psf_sh)
    psf_sh_all = np.asarray(psf_sh_all)
    
    # Add IPC
    if kipc is not None or ipc_vals is not None:
        # Build kernel if it wasn't already specified
        if kipc is None:
            if isinstance(ipc_vals, (tuple, list, np.ndarray)):
                a1, a2 = ipc_vals
            else:
                a1, a2 = ipc_vals, 0
            kipc = np.array([[a2,a1,a2], [a1,1-4*(a1+a2),a1], [a2,a1,a2]])
        psf_sh_all = add_ipc(psf_sh_all, kernel=kipc)
    

    # Reshape to grid
    # sh_grid = (len(yoff_pix), len(xoff_pix))
    # xoff_all = xoff_all.reshape(sh_grid)
    # yoff_all = yoff_all.reshape(sh_grid)

    return xoff_pix, yoff_pix, psf_sh_all

def crop_observation(im_full, ap, xysub, xyloc=None, delx=0, dely=0, 
                     shift_func=fourier_imshift, interp='cubic',
                     return_xy=False, **kwargs):
    """Crop around aperture reference location

    `xysub` specifies the desired crop size.
    if `xysub` is an array, dimension order should be [nysub,nxsub]

    `xyloc` provides a way to manually supply the central position. 
    Set `ap` to None will crop around `xyloc` or center of array.

    Provides an options to shift array by some offset before cropping
    to allow for sub-pixel shifting. To change integer crop positions,
    recommend using `xyloc` instead.

    Shift function can be fourier_imshfit, fshift, or cv_shift.
    The interp keyword only works for the latter two options.
    Consider 'lanczos' for cv_shift.

    """
        
    # xcorn_sci, ycorn_sci = ap.corners('sci')
    # xcmin, ycmin = (int(xcorn_sci.min()+0.5), int(ycorn_sci.min()+0.5))
    # xsci_arr = np.arange(1, im_full.shape[1]+1)
    # ysci_arr = np.arange(1, im_full.shape[0]+1)

    
    # Cut out postage stamp from full frame image
    if isinstance(xysub, (list, tuple, np.ndarray)):
        ny_sub, nx_sub = xysub
    else:
        ny_sub = nx_sub = xysub
    
    # Get centroid position
    if ap is None:
        xc, yc = get_im_cen(im_full) if xyloc is None else xyloc
    else: 
        # Subtract 1 from sci coords to get indices
        xc, yc = (ap.XSciRef-1, ap.YSciRef-1) if xyloc is None else xyloc

    x1 = round_int(xc - nx_sub/2 + 0.5)
    x2 = x1 + nx_sub
    y1 = round_int(yc - ny_sub/2 + 0.5)
    y2 = y1 + ny_sub

    # Save initial values in case they get modified below
    x1_init, x2_init = (x1, x2)
    y1_init, y2_init = (y1, y2)

    sh_orig = im_full.shape
    if (x2>sh_orig[0]) or (y2>sh_orig[1]) or (x1<0) or (y1<0):
        dx = x2 - sh_orig[0]
        dx = 0 if dx<0 else dx
        dy = y2 - sh_orig[1]
        dy = 0 if dy<0 else dy
        # Expand image
        shape_new = (2*dy+sh_orig[0], 2*dx+sh_orig[1])
        im_full = pad_or_cut_to_size(im_full, shape_new)

        xc_new, yc_new = (xc+dx, yc+dy)
        x1 = round_int(xc_new - nx_sub/2 + 0.5)
        x2 = x1 + nx_sub
        y1 = round_int(yc_new - ny_sub/2 + 0.5)
        y2 = y1 + ny_sub

    if (x1<0) or (y1<0):
        dx = -1*x1 if x1<0 else 0
        dy = -1*y1 if y1<0 else 0

        # Expand image
        shape_new = (2*dy+sh_orig[0], 2*dx+sh_orig[1])
        im_full = pad_or_cut_to_size(im_full, shape_new)

        xc_new, yc_new = (xc+dx, yc+dy)
        x1 = round_int(xc_new - nx_sub/2 + 0.5)
        x2 = x1 + nx_sub
        y1 = round_int(yc_new - ny_sub/2 + 0.5)
        y2 = y1 + ny_sub

    # Perform pixel shifting
    if delx!=0 or dely!=0:
        kwargs['interp'] = interp
        # Use fshift function if only performing integer shifts
        if float(delx).is_integer() and float(dely).is_integer():
            shift_func = fshift
        im_full = shift_func(im_full, delx, dely, **kwargs)
    
    im = im_full[y1:y2, x1:x2]
    xy_ind = np.array([x1_init, x2_init, y1_init, y2_init])
    
    if return_xy:
        return im, xy_ind
    else:
        return im

def get_im_cen(im):
    """
    Returns pixel position of array center.
    For odd dimensions, this is in a pixel center.
    For even dimensions, this is at the pixel boundary.
    """
    ny, nx = im.shape
    return np.array([nx / 2. - 0.5, ny / 2. - 0.5])

def crop_image(im, xysub, xyloc=None, **kwargs):
    """Crop input image around center using integer offsets only"""
    
    return crop_observation(im, None, xysub, xyloc=xyloc, **kwargs)


def find_offsets(input, psf, crop=65, xlim_pix=(-3,3), ylim_pix=(-3,3), 
    shift_func=fshift, rin=0, rout=None, dxy_coarse=0.05, dxy_fine=0.01):
    """Find offsets necessary to align observations with input psf"""
        
    # Check if input is a dictionary 
    is_dict = True if isinstance(input, dict) else False

    res = gen_psf_offsets(psf, crop=crop, xlim_pix=xlim_pix, ylim_pix=ylim_pix, 
                          dxy=dxy_coarse, shift_func=shift_func)
    xoff_pix, yoff_pix, psf_sh_all = res

    # Grid shape
    sh_grid = (len(yoff_pix), len(xoff_pix))

    # Cycle through each SGD position
    keys = list(input.keys()) if is_dict else None    

    xsh0_pix = []
    ysh0_pix = []
    iter_vals = tqdm(keys) if is_dict else tqdm(input)
    for val in iter_vals:
        if is_dict:
            d = input[val]
            im = crop_observation(d['data'], d['ap'], crop)
        else:
            im = pad_or_cut_to_size(val, crop)

        # Create masks
        rdist = dist_image(im)
        rin = 0 if rin is None else rin
        rmask = (rdist>=rin) if rout is None else (rdist>=rin) & (rdist<=rout)
        # Exclude 0s and NaNs
        zmask = (im!=0) & (~np.isnan(im))
        ind_mask = rmask & zmask

        # Cross-correlate to find best x,y shift to align image with PSF
        cc = correl_images(psf_sh_all, im, mask=ind_mask)
        cc = cc.reshape(sh_grid)
        
        # Cubic interplotion of cross correlation image onto a finer grid
        xsh, ysh = find_max_crosscorr(cc, xoff_pix, yoff_pix, dxy_fine)
        
        xsh0_pix.append(xsh)
        ysh0_pix.append(ysh)

    xsh0_pix = np.array(xsh0_pix)
    ysh0_pix = np.array(ysh0_pix)
    
    return xsh0_pix, ysh0_pix


def find_offsets2(input, xoff_pix, yoff_pix, psf_sh_all, bpmasks=None,
    crop=65, rin=0, rout=None, dxy_fine=0.01, prog_leave=True, **kwargs):
    """Find offsets necessary to align observations with input psf"""
        
    # Check if input is a dictionary 
    is_dict = True if isinstance(input, dict) else False
    
    # Make sure input image is 3D
    if not is_dict and len(input.shape)==2:
        input2d = True
        input = [input]
    else:
        input2d = False

    if (bpmasks is not None) and (len(bpmasks.shape)==2):
        bpmasks = [bpmasks]

    # Grid shape
    sh_grid = (len(yoff_pix), len(xoff_pix))

    # Cycle through each SGD position
    keys = list(input.keys()) if is_dict else None

    xsh0_pix = []
    ysh0_pix = []
    iter_vals = tqdm(keys,leave=prog_leave) if is_dict else tqdm(input,leave=prog_leave)
    i = 0
    for val in iter_vals:
        
        if crop is None:
            im0 = input[val]['data'] if is_dict else val
            ny1, nx1 = im0.shape
            _, ny2, nx2 = psf_sh_all
            ny_crop = np.min([ny1, ny2])
            nx_crop = np.min([nx1, nx2])
            crop = (ny_crop, nx_crop)

        # Crop the input image
        if is_dict:
            d = input[val]
            im = crop_observation(d['data'], d['ap'], crop)
        else:
            im = crop_image(val, crop)

        # Crop PSFs to match size
        psf_sh_crop = np.array([crop_image(psf, crop) for psf in psf_sh_all])

        # Crop bp mask to match 
        if bpmasks is None:
            bpmask = np.zeros_like(im).astype('bool')
        else:
            bpmask = crop_image(bpmasks[i], crop)
            i += 1

        # print(im.shape, psf_sh_crop.shape, psf_sh_all.shape)

        # Create masks
        rdist = dist_image(im)
        rin = 0 if rin is None else rin
        rmask = (rdist>=rin) if rout is None else (rdist>=rin) & (rdist<=rout)
        # Exclude 0s and NaNs
        zmask = (im!=0) & (~np.isnan(im))
        ind_mask = rmask & zmask & (~bpmask)
        
        # Cross-correlate to find best x,y shift to align image with PSF
        cc = correl_images(psf_sh_crop, im, mask=ind_mask)
        cc = cc.reshape(sh_grid)
        
        # Cubic interplotion of cross correlation image onto a finer grid
        xsh, ysh = find_max_crosscorr(cc, xoff_pix, yoff_pix, dxy_fine)
        
        xsh0_pix.append(xsh)
        ysh0_pix.append(ysh)

    xsh0_pix = np.array(xsh0_pix)
    ysh0_pix = np.array(ysh0_pix)

    # If we had a single image input, return first elements
    if input2d:
        xsh0_pix = xsh0_pix[0]
        ysh0_pix = ysh0_pix[0]
    
    return xsh0_pix, ysh0_pix


def find_offsets_phase(input, psf, crop=65, rin=0, rout=None, dxy_fine=0.01, 
    prog_leave=False):
    """Use phase_cross_correlation to determine offset 
    
    Returns offset (delx,dely) required to register input image[s] onto psf image.
    """

    # Check if input is a dictionary 
    is_dict = True if isinstance(input, dict) else False
    
    # Make sure input image is 3D
    if not is_dict and len(input.shape)==2:
        input = [input]

    # Cycle through each SGD position
    keys = list(input.keys()) if is_dict else None
    
    # Ensure PSF is correct size
    psf_sub = pad_or_cut_to_size(psf, crop)

    xsh0_pix = []
    ysh0_pix = []
    if prog_leave:
        iter_vals = tqdm(keys) if is_dict else tqdm(input)
    else:
        iter_vals = keys if is_dict else input
    for val in iter_vals:
        if is_dict:
            d = input[val]
            imfull = d['data']
            im = crop_observation(imfull, d['ap'], crop).copy()
        else:
            imfull = val
            im = pad_or_cut_to_size(imfull, crop)

        # Create masks
        rdist = dist_image(im)
        rin = 0 if rin is None else rin
        rmask = (rdist>=rin) if rout is None else (rdist>=rin) & (rdist<=rout)
        # Exclude 0s and NaNs
        zmask = (im!=0) & (~np.isnan(im))
        ind_mask = rmask & zmask
        
        # Zero-out bad pixels
        im[~ind_mask] = 0

        # Initial offset required to move im onto psf_sub
        ysh, xsh = phase_cross_correlation(psf_sub, im, upsample_factor=1/dxy_fine, 
                                           return_error=False)
        
        # Shift PSF in opposite direction to register onto im.
        # We do this under the assumption that PSF is more ideal (no bad pixels) compared to im,
        # so there will less fourier artifacts after the shift.
        # Then find any residual necessary moves.
        psf_sh = pad_or_cut_to_size(fourier_imshift(psf, -1*xsh, -1*ysh), crop)
        del_ysh, del_xsh = phase_cross_correlation(psf_sh, im, upsample_factor=1/dxy_fine, 
                                                   return_error=False)
        xsh += del_xsh
        ysh += del_ysh
        
        xsh0_pix.append(xsh)
        ysh0_pix.append(ysh)

    xsh0_pix = np.array(xsh0_pix)
    ysh0_pix = np.array(ysh0_pix)
    
    res = np.array([xsh0_pix, ysh0_pix]).T
    
    return res.squeeze()


def get_com(im, halfwidth=7, return_sci=False, **kwargs):
    
    from poppy.fwcentroid import fwcentroid
    
    # Find center of mass centroid
    com = fwcentroid(im, halfwidth=halfwidth, **kwargs)
    yind_com, xind_com = com
    
    if return_sci:
        return xind_com+1, yind_com+1
    else:
        return xind_com, yind_com

def recenter_psf(psfs_over, niter=3, halfwidth=7):
    """Use center of mass algorithm to relocate PSF to center of image.
    
    Returns recentered PSFs and shift values used.
    """

    from webbpsf_ext.image_manip import fourier_imshift

    ndim = len(psfs_over.shape)
    if ndim==2:
        psfs_over = [psfs_over]

    # Reposition oversampled PSF to center of array using center of mass algorithm
    xyoff_psfs_over = []
    for i, psf in enumerate(psfs_over):
        xc_psf, yc_psf = get_im_cen(psf)
        xsh_sum, ysh_sum = (0, 0)
        for j in range(niter):
            xc, yc = get_com(psf, halfwidth=halfwidth, return_sci=False)
            xsh, ysh = (xc_psf - xc, yc_psf - yc)
            psf = fourier_imshift(psf, xsh, ysh)
            xsh_sum += xsh
            ysh_sum += ysh
        psfs_over[i] = psf
        xyoff_psfs_over.append(np.array([xsh_sum, ysh_sum]))
        
    # Oversampled offsets
    xyoff_psfs_over = np.array(xyoff_psfs_over)

    # If input was a single image, return same dimensions
    if ndim==2:
        psfs_over = psfs_over[0]
        xyoff_psfs_over = xyoff_psfs_over[0]

    return psfs_over, xyoff_psfs_over
