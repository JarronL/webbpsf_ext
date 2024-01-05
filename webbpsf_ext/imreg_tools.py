import numpy as np
import os
from tqdm.auto import tqdm

from .image_manip import fourier_imshift, fshift, frebin
from .image_manip import get_im_cen, pad_or_cut_to_size, bp_fix
from .coords import dist_image, get_sgd_offsets
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
#    File Information
###########################################################################

def get_detname(det_id, use_long=True):
    """Return NRC[A-B][1-4,LONG] for valid detector/SCA IDs"""

    from .utils import get_detname as _get_detname
    return _get_detname(det_id, use_long=use_long)

def get_mask_from_pps(apname_pps):
    """Get mask name from PPS aperture name
    
    The PPS aperture name is of the form:
        NRC[A/B][1-5]_[FULL]_[MASK]
    where MASK is the name of the coronagraphic mask used.

    For target acquisition apertures the mask name can be
    prependend with "TA" (eg., TAMASK335R).

    Return '' if MASK not in input aperture name.
    """

    if 'MASK' not in apname_pps:
        return ''

    pps_str_arr = apname_pps.split('_')
    for s in pps_str_arr:
        if 'MASK' in s:
            image_mask = s
            break

    # Special case for TA apertures
    if 'TA' in image_mask:
        # Remove TA from mask name
        image_mask = image_mask.replace('TA', '')

        # Remove FS from mask name
        if 'FS' in image_mask:
            image_mask = image_mask.replace('FS', '')

        # Remove trailing S or L from LWB and SWB TA apertures
        if ('WB' in image_mask) and (image_mask[-1]=='S' or image_mask[-1]=='L'):
            image_mask = image_mask[:-1]

    return image_mask

def get_coron_apname(input):
    """Get aperture name from header or data model
    
    Parameters
    ==========
    input : fits.header.Header or datamodels.DataModel
        Input header or data model
    """

    if isinstance(input, (fits.header.Header)):
        # Aperture names
        apname = input['APERNAME']
        apname_pps = input['PPS_APER']
        subarray = input['SUBARRAY']
    else:
        # Data model meta info
        meta = input.meta

        # Aperture names
        apname = meta.aperture.name
        apname_pps = meta.aperture.pps_name
        subarray = meta.subarray.name

    # print(apname, apname_pps, subarray)

    # No need to do anything if the aperture names are the same
    # Also skip if MASK not in apname_pps
    if ((apname==apname_pps) or ('MASK' not in apname_pps)) and ('400X256' not in subarray):
        apname_new = apname
    else:
        # Should only get here if coron mask and apname doesn't match PPS
        apname_str_split = apname.split('_')
        sca = apname_str_split[0]
        image_mask = get_mask_from_pps(apname_pps)

        # Get subarray info
        # Sometimes apname erroneously has 'FULL' in it
        # So, first for subarray info in apname_pps 
        if ('400X256' in apname_pps) or ('400X256' in subarray):
            apn0 = f'{sca}_400X256'
        elif ('FULL' in apname_pps):
            apn0 = f'{sca}_FULL'
        else:
            apn0 = sca

        apname_new = f'{apn0}_{image_mask}'

        # Append filter or NARROW if needed
        pps_str_arr = apname_pps.split('_')
        last_str = pps_str_arr[-1]
        # Look for filter specified in PPS aperture name
        if ('_F1' in apname_pps) or ('_F2' in apname_pps) or ('_F3' in apname_pps) or ('_F4' in apname_pps):
            # Find all instances of "_"
            inds = [pos for pos, char in enumerate(apname_pps) if char == '_']
            # Filter is always appended to end, but can have different string sizes (F322W2)
            filter = apname_pps[inds[-1]+1:]
            apname_new += f'_{filter}'
        elif last_str=='NARROW':
            apname_new += '_NARROW'
        elif ('TAMASK' in apname_pps) and ('WB' in apname_pps[-1]):
            apname_new += '_WEDGE_BAR'
        elif ('TAMASK' in apname_pps) and (apname_pps[-1]=='R'):
            apname_new += '_WEDGE_RND'

    # print(apname_new)

    # If apname_new doesn't exit, we need to fall back to apname
    # even if it may not fully make sense.
    if apname_new in nrc_siaf.apernames:
        return apname_new
    else:
        return apname

def apname_full_frame_coron(apname):
    """Retrieve full frame version of coronagraphic aperture name"""

    if 'FULL' in apname:
        if 'FULL_WEDGE' in apname:
            _log.warning(f'Aperture name {apname} does not specify occulting mask.')
        return apname
    else:
        # Remove 400X256 string
        apname = apname.replace('_400X256', '')
        # Add in FULL string
        apname_full = apname.replace('_', '_FULL_', 1)
        return apname_full

def get_files(indir, pid, obsid=None, sca=None, filt=None, file_type='uncal.fits', 
              exp_type=None, vst_grp_act=None, apername=None, apername_pps=None):
    """Get files of interest
    
    Parameters
    ==========
    indir : str
        Location of FITS files.
    obsid : int
        Observation number.
    sca : str
        Name of detector (e.g., 'along' or 'a3')
    filt : str
        Return files observed in given filter.
    file_type : str
        uncal.fits or rate.fits, etc
    exp_type : str
        Exposure type such as NRC_TACQ, NRC_TACONFIRM, NRC_CORON, etc.
    vst_grp_act : str
        The _<gg><s><aa>_ portion of the file name.
        hdr0['VISITGRP'] + hdr0['SEQ_ID'] + hdr0['ACT_ID']
    apername : str
        Name of aperture (e.g., NRCA5_FULL)
    apername_pps : str
        Name of aperture from PPS (e.g., NRCA5_FULL)
    """

    sca = '' if sca is None else get_detname(sca).lower()
    
    # file name start and end
    if obsid is None:
        file_start = f'jw{pid:05d}'
    else:
        file_start = f'jw{pid:05d}{obsid:03d}'
    sca = sca.lower()

    if file_type[0]=='_':
        file_type = file_type[1:]
    file_end = f'{sca}_{file_type}'

    allfiles = np.sort([f for f in os.listdir(indir) if ((file_end in f) and f.startswith(file_start))])
    
    # Check filter info
    if filt is not None:
        files2 = []
        for f in allfiles:
            hdr = fits.getheader(os.path.join(indir,f))
            obs_filt = hdr.get('FILTER', 'none')
            obs_pup = hdr.get('PUPIL', 'none')
            # Check if filter string exists in the pupil wheel
            if obs_pup[0]=='F' and (obs_pup[-1]=='N' or obs_pup[-1]=='M'):
                filt_match = obs_pup
            else:
                filt_match = obs_filt
            if filt==filt_match:
                files2.append(f)
        allfiles = np.array(files2)

    # Filter by exposure type
    if exp_type is not None:
        files2 = []
        for f in allfiles:
            hdr = fits.getheader(os.path.join(indir,f))
            if hdr.get('EXP_TYPE', 'none')==exp_type:
                files2.append(f)
        allfiles = np.array(files2)

    # Filter by visit group
    if vst_grp_act is not None:
        files2 = []
        for f in allfiles:
            hdr = fits.getheader(os.path.join(indir,f))
            if hdr.get('VISITGRP', 'none')==vst_grp_act[0:2].upper() and \
               hdr.get('SEQ_ID', 'none')==vst_grp_act[2].upper() and \
               hdr.get('ACT_ID', 'none')==vst_grp_act[3:].upper():
                # print(f)
                files2.append(f)
        allfiles = np.array(files2)

    if apername is not None:
        files2 = []
        for f in allfiles:
            hdr = fits.getheader(os.path.join(indir,f))
            apname_obs = hdr.get('APERNAME', 'none')
            if apname_obs==apername or apername==get_coron_apname(hdr):
                files2.append(f)
        allfiles = np.array(files2)

    if apername_pps is not None:
        files2 = []
        for f in allfiles:
            hdr = fits.getheader(os.path.join(indir,f))
            apname_pps = hdr.get('PPS_APER', 'none')
            if apname_pps==apername_pps:
                files2.append(f)
        allfiles = np.array(files2)
    
    return allfiles

def filter_files(files, save_dir):
    """Remove files where source is offset off of observed aperture"""

    # Check if we've dithered outside of FoV
    exp_ind = get_loc_all(files, save_dir, find_func=get_expected_loc)

    ind_keep = []
    for i, f in enumerate(files):
        xi, yi = exp_ind[i]
        
        # Open FITS file
        fpath = os.path.join(save_dir, f)
        hdul = fits.open(fpath)
        hdr = hdul[0].header
        ap = nrc_siaf[hdr['APERNAME']]

        if (0<xi<ap.XSciSize) and (0<yi<ap.YSciSize):
            ind_keep.append(i)

        # Close FITS file
        hdul.close()

    return files[ind_keep]

def get_save_dir(pid):
    """Return save directory for processed files
    
    Takes the MAST directory for a given PID and adds
    '_proc' to the end. If it doesn't exist, it will be created.
    """
    mast_dir = os.getenv('JWSTDOWNLOAD_OUTDIR')

    # Output directory
    if mast_dir[-1]=='/':
        mast_proc_dir = mast_dir[:-1] + '_proc/'
    else:
        mast_proc_dir = mast_dir + '_proc/'
    save_dir = os.path.join(mast_proc_dir, f'{pid:05d}/')

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    return save_dir

def load_cropped_files(save_dir, files, xysub=65, bgsub=False, 
                       fix_bad_pixels=True, **kwargs):

    from jwst.datamodels import dqflags

    # Get index location and 'sci' position
    halfwidth = kwargs.get('halfwidth', 15)
    com_ind = get_loc_all(files, save_dir, find_func=get_com,
                          fix_bad_pixels=fix_bad_pixels, 
                          halfwidth=halfwidth)

    imsub_arr = []
    dqsub_arr = []
    xyind_arr = []
    for i, f in enumerate(files):
        fpath = os.path.join(save_dir, f)
        hdul = fits.open(fpath)

        # Crop and roughly center image
        data, xy = crop_image(hdul['SCI'].data, xysub, xyloc=com_ind[i], return_xy=True)
        try:
            dqmask = crop_image(hdul['DQ'].data, xysub, xyloc=com_ind[i])
        except KeyError:
            dqmask = np.zeros_like(data).astype(np.uint64)
        
        # For arrays padded with 0s, flag those pixels as DO_NOT_USE
        indz = (data==0)
        dqmask[indz] = dqmask[indz] | dqflags.pixel['DO_NOT_USE']
        
        imsub_arr.append(data)
        dqsub_arr.append(dqmask)
        xyind_arr.append(xy)
        
        hdul.close()

    imsub_arr = np.array(imsub_arr)
    dqsub_arr = np.array(dqsub_arr)
    xyind_arr = np.array(xyind_arr)
    bp_masks1 = (dqsub_arr & dqflags.pixel['OTHER_BAD_PIXEL']) > 0
    bp_masks2 = (dqsub_arr & dqflags.pixel['DO_NOT_USE']) > 0
    bp_masks = bp_masks1 | bp_masks2 | np.isnan(imsub_arr)

    # Do bg subtraction from r>bg_rad and only include good pixels
    if bgsub:
        # Radial position to set background
        bg_rad = int(0.7 * xysub / 2)
        ind_bg = dist_image(np.zeros([xysub,xysub])) > bg_rad
        for i in range(len(files)):
            indgood = (~bp_masks[i]) & ind_bg
            imsub_arr[i] -= np.nanmedian(data[indgood])

    return imsub_arr, dqsub_arr, xyind_arr, bp_masks

###########################################################################
#    Target Acquisition
###########################################################################

def get_ictm_event_log(startdate, enddate, hdr=None, mast_api_token=None, verbose=False):
    """Get ICTM event log from MAST
    
    Parameters
    ==========
    startdate : str
        Start date of observation of format YYYY-MM-DD HH:MM:SS.sss
    enddate : str
        End date of observation of format YYYY-MM-DD HH:MM:SS.sss
    """

    from datetime import datetime, timedelta, timezone
    from requests import Session
    import time

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
    try:
        idot = startdate.index('.')
        startdate = startdate[0:idot]
    except ValueError:
        pass

    enddate = enddate.replace(' ', '+')
    try:
        idot = enddate.index('.')
        enddate = enddate[0:idot]
    except ValueError:
        pass

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

    retries = 0
    retry_limit = 5
    while retries < retry_limit:
        try:
            response.raise_for_status()
            break
        except Exception as e:
            # Wait 5 seconds before retrying
            time.sleep(5)
            # log the error
            retries += 1
            if retries == retry_limit:
                _log.error(f'Failed to retreieve url after {retry_limit} tries')
                raise e

    lines = response.content.decode('utf-8').splitlines()

    return lines

def tasub_to_apname(tasub):

    # Get aperture name from TA subarray name

    # Dictionary of aperture names
    apname_dict={
        'SUBFSA210R' : 'NRCA2_FSTAMASK210R' ,
        'SUBFSA335R' : 'NRCA5_FSTAMASKM335R',
        'SUBFSA430R' : 'NRCA5_FSTAMASKM430R',
        'SUBFSALWB'  : 'NRCA5_FSTAMASKLWB'  ,
        'SUBFSASWB'  : 'NRCA4_FSTAMASKSWB'  ,
        'SUBNDA210R' : 'NRCA2_TAMASK210R'   ,
        'SUBNDA335R' : 'NRCA5_TAMASK335R'   ,
        'SUBNDA430R' : 'NRCA5_TAMASK430R'   ,
        'SUBNDALWBL' : 'NRCA5_TAMASKLWBL'   ,
        'SUBNDALWBS' : 'NRCA5_TAMASKLWB'    ,
        'SUBNDASWBL' : 'NRCA4_TAMASKSWB'    ,
        'SUBNDASWBS' : 'NRCA4_TAMASKSWBS'   ,
        'SUBNDB210R' : 'NRCB1_TAMASK210R'   ,
        'SUBNDB335R' : 'NRCB5_TAMASK335R'   ,
        'SUBNDB430R' : 'NRCB5_TAMASK430R'   ,
        'SUBNDBLWBL' : 'NRCB5_TAMASKLWBL'   ,
        'SUBNDBLWBS' : 'NRCB5_TAMASKLWB'    ,
        'SUBNDBSWBL' : 'NRCB3_TAMASKSWB'    ,
        'SUBNDBSWBS' : 'NRCB3_TAMASKSWBS'   ,
    }

    return apname_dict[tasub]



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
        val_str = value[2]

        # Get subarray name for visit
        if in_selected_visit and  'Configured NIRCam subarray' in val_str:
            val_str_list = val_str.split(' ')
            tasub = val_str_list[-1].split(',')[0]
            _log.info(val_str)
            
        if in_selected_visit and ((not ta_only) or in_ta) :
            # print(value[0][0:22], "\t", value[2])
            
            # Print coordinate location info
            if ('postage-stamp coord' in val_str) or ('detector coord' in val_str): 
                _log.info(val_str)
        
            # Backup coords in case of TA centroid failure
            if 'postage-stamp coord (colPeak, rowPeak)' in val_str:
                val_str_list = val_str.split('=')
                xcen, ycen = val_str_list[1].split(',')
                ind1 = xcen.find('(')
                xcen = xcen[ind1+1:]
                ind2 = ycen.find(')')
                ycen = ycen[0:ind2]
                # These are NOT 'sci' coords, but instead a 
                # subarray cut-out in detector coords
                peak_coords = (float(xcen), float(ycen))

            # Parse centroid position reported in detector coordinates
            if 'detector coord (colCentroid, rowCentroid)' in val_str:
                val_str_list = val_str.split('=')
                xcen, ycen = val_str_list[1].split(',')
                ind1 = xcen.find('(')
                xcen = xcen[ind1+1:]
                ind2 = ycen.find(')')
                ycen = ycen[0:ind2]

                return float(xcen), float(ycen)

        # Flag if current line is between when visit starts and ends
        if val_str[:6] == 'VISIT ':
            if val_str[-7:] == 'STARTED':
                vstart = 'T'.join(value[0].split())[:-3]
                vid = val_str.split()[1]

                if vid==selected_visit_id:
                    _log.debug(f"VISIT {selected_visit_id} START FOUND at {vstart}")
                    in_selected_visit = True
                    # if ta_only:
                    #     print("Only displaying TARGET ACQUISITION RESULTS:")

            elif val_str[-5:] == 'ENDED' and in_selected_visit:
                assert vid == val_str.split()[1]
                assert selected_visit_id  == val_str.split()[1]

                vend = 'T'.join(value[0].split())[:-3]
                _log.debug(f"VISIT {selected_visit_id} END FOUND at {vend}")

                in_selected_visit = False
        elif val_str[:31] == f'Script terminated: {vid}':
            if val_str[-5:] == 'ERROR':
                script = val_str.split(':')[2]
                vend = 'T'.join(value[0].split())[:-3]
                dur = datetime.fromisoformat(vend) - datetime.fromisoformat(vstart)
                note = f'Halt in {script}'
                in_selected_visit = False
        elif in_selected_visit and val_str.startswith('*'): 
            # this string is used to mark the start and end of TA sections
            in_ta = not in_ta

    # If we've gotten here, then no centroid was found
    # Return peak coords if available
    if 'peak_coords' in locals():
        _log.warning(f'No centroid found for {selected_visit_id}. Using peak coords instead.')
        apname = tasub_to_apname(tasub)
        ap = nrc_siaf[apname]
        x0, y0 = np.min(ap.corners('det'), axis=1)

        # Figure out location of peak in full frame
        xp_full = peak_coords[0] + x0 - 0.5
        yp_full = peak_coords[1] + y0 - 0.5

        return np.array([xp_full, yp_full])
    else:
        _log.warning(f'No centroid found for {selected_visit_id}.')
        return None

def diff_ta_data(uncal_data):
    """Onboard algorithm to difference TA data"""
    
    data = uncal_data.astype('float')
    
    nint, ng, ny, nx = data.shape
    im1 = data[0,-1]    - data[0,ng//2]
    im2 = data[0,ng//2] - data[0,0]
    
    return np.minimum(im1,im2)

def read_ta_files(indir, pid, obsid, sca, file_type='rate.fits', 
                  uncal_dir=None, bpfix=False):
    """Store all TA and Conf data into a dictionary
    
    indir should include rate.fits. For the initial TACQ, can use
    uncal files (via `uncal_dir` input flag) to simulate onboard
    subtraction. 

    bpfix is only for the TACONF data and mainly for display purposes.

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
    file_type : str
        File extension, such as uncal.fits, rate.fits, cal.fits, etc.
    uncal_dir : str or None
        If not None, use uncal files in this directory for TACQ data.
    bpfix : bool
        If True, perform bad pixel fixing on the data.
        Mainly for display purposes.
    """

    from jwst import datamodels

    # Option to use uncal files for subarray TA observation
    ta_dir = indir if uncal_dir is None else uncal_dir
    taconf_dir = indir

    fta_type = file_type if uncal_dir is None else 'uncal.fits'

    # Get TACQ
    try:
        fta = get_files(ta_dir, pid, obsid=obsid, sca=sca, 
                        file_type=fta_type, exp_type='NRC_TACQ')[0]
    except:
        raise RuntimeError(f'Unable to determine NRC_TACQ file for PID {pid} Obs, {obsid}, {sca}')

    # Full path
    fta_path = os.path.join(ta_dir, fta)
    ta_dict = {'dta': {'file': fta_path, 'type': 'Target Acq'}}

    # Get TACONFIRM 
    fconf = get_files(taconf_dir, pid, obsid=obsid, sca=sca, 
                      file_type=file_type, exp_type='NRC_TACONFIRM')
    if len(fconf)>0:
        fconf1, fconf2 = fconf
        # Full paths of files
        fconf1_path = os.path.join(taconf_dir, fconf1)
        fconf2_path = os.path.join(taconf_dir, fconf2)

        ta_dict['dconf1'] = {'file': fconf1_path, 'type': 'TA Conf1'}
        ta_dict['dconf2'] = {'file': fconf2_path, 'type': 'TA Conf2'}
    else:
        _log.warning(f'NRC_TACQ exists, but no NRC_TACONFIRM observed for PID {pid}, Obs {obsid}, {sca}')

    # Build dictionary of data and header info
    for k in ta_dict.keys():
        d = ta_dict[k]

        f = d['file']
        # print(f)
        hdul = fits.open(f)
        # Get data and take diff if uncal
        data = hdul['SCI'].data.astype('float')
        if 'uncal.fits' in f:
            # For TACQ, do difference and get DQ mask from rate file
            data = diff_ta_data(data)
            frate = get_files(indir, pid, obsid=obsid, sca=sca, 
                              file_type=file_type, exp_type='NRC_TACQ')[0]
            frate_path = os.path.join(indir, frate)
            dq = fits.getdata(frate_path, extname='DQ')
        else:
            dq = hdul['DQ'].data

        # Get date from datamodel
        data_model = datamodels.open(f)
        date = data_model.meta.observation.date_beg
        # Close data model
        data_model.close()

        d['data'] = data
        d['dq']   = dq
        d['hdr0'] = hdul[0].header
        d['hdr1'] = hdul[1].header
        d['date'] = date
        hdul.close()

        d['apname'] = get_coron_apname(d['hdr0'])
        # Apername supplied by PPS for pointing control
        d['apname_pps'] = d['hdr0']['PPS_APER']

        d['ap'] = nrc_siaf[d['apname']]
        d['ap_pps'] = nrc_siaf[d['apname_pps']]

        # Exposure type
        d['exp_type'] = d['hdr0']['EXP_TYPE']

        # bad pixel fixing for TA confirmation
        if bpfix and ('conf' in k):
            im = crop_observation(d['data'], d['ap'], 100)
            # Perform pixel fixing in place
            _ = bp_fix(im, sigclip=10, niter=1, in_place=True)
        
    return ta_dict


def read_sgd_files(indir, pid, obsid, filter, sca, bpfix=False, 
                   file_type='rate.fits', exp_type=None, vst_grp_act=None,
                   apername=None, apername_pps=None, nodata=False):
    """Store SGD or science data into a dictionary

    By default, excludes any TAMASK or TACONFIRM data, but can be overridden
    by setting exp_type.
    
    Parameters
    ==========
    indir : str
        Input directory
    pid : int
        Program ID number
    obsid : int
        Observation number
    filter : str
        Name of filter element
    sca : str
        SCA name, such as a1, a2, a3, a4, along, etc
    file_type : str
        File extension, such as uncal.fits, rate.fits, cal.fits, etc.
    exp_type : str
        Exposure type such as NRC_TACQ, NRC_TACONFIRM
    vst_grp_act : str
        The _<gg><s><aa>_ portion of the file name.
        hdr0['VISITGRP'] + hdr0['SEQ_ID'] + hdr0['ACT_ID']
    apername : str
        Name of aperture (e.g., NRCA5_FULL)
    apername_pps : str
        Name of aperture from PPS (e.g., NRCA5_FULL)
    bpfix : bool
        If True, perform bad pixel fixing on the data.
        Mainly for display purposes.
    nodata : bool
        If True, only return header info and not data.
    """

    from jwst import datamodels

    files = get_files(indir, pid, obsid=obsid, sca=sca, filt=filter,
                      file_type=file_type, exp_type=exp_type, vst_grp_act=vst_grp_act,
                      apername=apername, apername_pps=apername_pps)

    # Exclude any TAMASK or TACONFIRM data by default
    if exp_type is None:
        ikeep = []
        for i, f in enumerate(files):
            fpath = os.path.join(indir, f)
            hdr = fits.getheader(fpath, ext=0)
            isTA = ('_TACQ' in hdr['EXP_TYPE']) or ('_TACONFIRM' in hdr['EXP_TYPE'])
            if not isTA:
                ikeep.append(i)
        
        files = files[ikeep]

    sgd_dict = {}
    for i, f in enumerate(files):
        fpath = os.path.join(indir, f)
        d = {'file': fpath}
        
        hdul = fits.open(fpath)
        if not nodata:
            d['data'] = hdul['SCI'].data.astype('float')
        d['dq']   = hdul['DQ'].data
        d['hdr0'] = hdul[0].header
        d['hdr1'] = hdul[1].header
        hdul.close()

        # Get date from datamodel
        data_model = datamodels.open(fpath)
        d['date'] = data_model.meta.observation.date_beg
        # Close data model
        data_model.close()
 
        d['apname'] = get_coron_apname(d['hdr0'])
        # Apername supplied by PPS for pointing control
        d['apname_pps'] = d['hdr0']['PPS_APER']

        # Add SIAF apertures
        d['ap'] = nrc_siaf[d['apname']]
        d['ap_pps'] = nrc_siaf[d['apname_pps']]

        # Exposure type
        d['exp_type'] = d['hdr0']['EXP_TYPE']

        sgd_dict[i] = d

        # bad pixel fixing 
        if bpfix and not nodata:
            im = crop_observation(d['data'], d['ap'], 100)
            # Perform pixel fixing in place
            _ = bp_fix(im, sigclip=10, niter=1, in_place=True)

    return sgd_dict


###########################################################################
#    Image Cropping
###########################################################################

def get_expected_loc(input, return_indices=True, add_sroffset=None):
    """Input header or data model to get expected pixel position of target
    
    Integer values correspond to center of a pixel, whereas 0.5
    correspond to pixel edges.

    `return_indices=True` will return the [xi,yi] index within the 
    observed aperture subarray, otherwise returns the 'sci' coordinate 
    position. These should only be off 1 (e.g. index=sci-1, because
    'sci' coordinates are 1-index, while numpy arrays are 0-indexed).

    SR offsets excluded for dates prior to 2022-07-01, otherwise included.
    Specify `add_sroffset=True` or `add_sroffset=False` to override the
    default settings. If False, any SGD offsets will be added back in.
    TODO: What about normal dithers?

    Parameters
    ==========
    input : fits.header.Header or datamodels.DataModel
        Input header or data model
    return_indices : bool
        Return indices of expected location within the subarray
        otherwise return the 'sci' coordinate position.
    add_sroffset : None or bool
        Include Special Requirements (SR) offset in the calculation.
        Will default to False if date<2022-07-01, otherwise True.
        Specify True or False to override the default.
        If False, any SGD offsets will be added back in.
    """
    
    from astropy.time import Time

    apname = get_coron_apname(input)

    if isinstance(input, (fits.header.Header)):
        # Aperture names
        apname_pps = input['PPS_APER']
        # Dither offsets
        xoff_asec = input['XOFFSET']
        yoff_asec = input['YOFFSET']
        # date
        date_obs = input['DATE-OBS']

        # SGD info (only needed if SR offsets is False)
        is_sgd = input.get('SUBPXPAT', False)
        sgd_pattern = input.get('SMGRDPAT', None)
        sgd_pos = input.get('PATT_NUM', 1) - 1
    else:
        # Data model meta info
        meta = input.meta

        # Aperture names
        apname_pps = meta.aperture.pps_name
        # Dither offsets
        xoff_asec = meta.dither.x_offset
        yoff_asec = meta.dither.y_offset
        # date
        date_obs = meta.observation.date

        # SGD info (only needed if SR offsets is False)
        if hasattr(meta.dither, 'subpixel_pattern'):
            subpixel_pattern = meta.dither.subpixel_pattern
            if subpixel_pattern is None:
                is_sgd = False
            elif 'small-grid' in subpixel_pattern.lower():
                is_sgd = True
            else:
                is_sgd = False
        else:
            is_sgd = False
        # SGD type
        if is_sgd and hasattr(meta.dither, 'small_grid_pattern'):
            sgd_pattern = meta.dither.small_grid_pattern
        else:
            sgd_pattern = None
        # SGD position index
        if is_sgd and hasattr(meta.dither, 'position_number'):
            sgd_pos = meta.dither.position_number - 1
        else:
            sgd_pos = 0

    # Include SIAF subarray offset?
    # Set defaults
    if add_sroffset is None:
        # If observed before 2022-07-01, then don't include SR offset.
        # SR offsets prior to 2022-07-01 were included to match expected
        # changes to the SIAF that were made after July 1 (or around there).
        add_sroffset = False if Time(date_obs) < Time('2022-07-01') else True

    # If offsets excluded, then reset xoff and yoff to 0
    # but add in SGD offsets if they exist
    if not add_sroffset:
        xoff_asec = yoff_asec = 0.0
        # Add in a SGD offsets if they exist
        if is_sgd:
            xoff_arr, yoff_arr = get_sgd_offsets(sgd_pattern)
            xoff_asec += xoff_arr[sgd_pos]
            yoff_asec += yoff_arr[sgd_pos]

    # Observed aperture
    ap = nrc_siaf[apname]
    # Aperture reference for pointing / dithering
    ap_pps = nrc_siaf[apname_pps]
    
    # Expected pixel location based on ideal offset
    if apname == apname_pps:
        xsci_exp, ysci_exp = (ap.XSciRef, ap.YSciRef)
        # Add offset
        xsci_exp = xsci_exp + xoff_asec / ap.XSciScale
        ysci_exp = ysci_exp + yoff_asec / ap.YSciScale
    else:
        if np.allclose([xoff_asec, yoff_asec], 0.0):
            xtel, ytel = (ap_pps.V2Ref, ap_pps.V3Ref)
        else:
            xtel, ytel = ap_pps.idl_to_tel(xoff_asec, yoff_asec)
        xsci_exp, ysci_exp = ap.tel_to_sci(xtel, ytel)
    
    if return_indices:
        return xsci_exp-1, ysci_exp-1
    else:
        return xsci_exp, ysci_exp

def get_gfit_cen(im, xysub=11, return_sci=False, find_max=True):
    """Gaussion fit to get centroid position"""
    
    from astropy.modeling import models, fitting

    # Set NaNs to 0
    ind_nan = np.isnan(im)
    im[ind_nan] = 0

    # Crop around max value?
    if find_max:
        yind, xind = np.unravel_index(np.argmax(im), im.shape)
        xyloc = (xind, yind)
    else:
        xyloc = None
    im_sub, (x1, x2, y1, y2) = crop_image(im, xysub, return_xy=True, xyloc=xyloc)

    # Add crop indices create grid in terms of full image indices
    xv = np.arange(x1, x2)
    yv = np.arange(y1, y2)
    xgrid, ygrid = np.meshgrid(xv, yv)
    xc, yc = (xv.mean(), yv.mean())

    # Fit the data using astropy.modeling
    p_init = models.Gaussian2D(amplitude=im_sub.max(), x_mean=xc, y_mean=yc, x_stddev=1, y_stddev=2)
    fit_p = fitting.LevMarLSQFitter()

    pfit = fit_p(p_init, xgrid, ygrid, im_sub)
    xind_cen = pfit.x_mean.value
    yind_cen = pfit.y_mean.value

    # Return to NaNs
    im[ind_nan] = np.nan

    if return_sci:
        return xind_cen+1, yind_cen+1
    else:
        return xind_cen, yind_cen

def get_com(im, halfwidth=7, return_sci=False, **kwargs):
    """Center of mass centroiding"""
    
    from poppy.fwcentroid import fwcentroid

    # Set NaNs to 0
    ind_nan = np.isnan(im)
    im[ind_nan] = 0

    # Find center of mass centroid
    try:
        com = fwcentroid(im, halfwidth=halfwidth, **kwargs)
    except IndexError:
        hw = int(halfwidth / 2)
        com = fwcentroid(im, halfwidth=hw, **kwargs)
    yind_com, xind_com = com

    # Return to NaNs
    im[ind_nan] = np.nan

    if return_sci:
        return xind_com+1, yind_com+1
    else:
        return xind_com, yind_com

def get_loc_all(files, indir, find_func=get_com, 
                fix_bad_pixels=True, **kwargs):

    from jwst.datamodels import dqflags
    from .image_manip import bp_fix
    
    star_locs = []
    for f in files:
        fpath = os.path.join(indir, f)
        
        # Open FITS file
        hdul = fits.open(fpath)

        # Crop and roughly center image
        data = hdul['SCI'].data
        try:
            dqmask = hdul['DQ'].data
        except KeyError:
            dqmask = np.zeros_like(data).astype(np.uint64)

        # Get rough stellar position
        if find_func is get_expected_loc:
            xy = get_expected_loc(hdul[0].header, **kwargs)
        elif find_func is get_com:
            # Fix bad pixels
            bp_mask1 = (dqmask & dqflags.pixel['OTHER_BAD_PIXEL']) > 0
            bp_mask2 = (dqmask & dqflags.pixel['DO_NOT_USE']) > 0
            bpmask = bp_mask1 | bp_mask2

            if fix_bad_pixels:
                data = bp_fix(data, sigclip=20, in_place=False)
                data = bp_fix(data, bpmask=bpmask)
            else:
                data[bpmask] = np.nan

            xy = get_com(data, **kwargs)
        else:
            xy = find_func(data, **kwargs)
        
        star_locs.append(xy)
        
        # Close FITS file
        hdul.close()
        
    return np.array(star_locs)

def recenter_psf(psfs_over, niter=3, halfwidth=7, 
                 gfit=True, in_place=False, **kwargs):
    """Use center of mass algorithm to relocate PSF to center of image.
    
    Returns recentered PSFs and shift values used.

    Parameters
    ----------
    psfs_over : array_like
        Oversampled PSF(s) to recenter. If 2D, will be converted to 3D.
    niter : int
        Number of iterations to use for center of mass algorithm.
    halfwidth : int or None
        Halfwidth of box to use for center of mass algorithm.
        Default is 7, which is a 15x15 box.
    gfit : bool
        If True, use Gaussian fitting instead of center of mass.
    in_place : bool
        If True, then perform the shift in place, overwriting the input
        PSF array.
    """

    from .image_manip import fourier_imshift

    ndim = len(psfs_over.shape)
    if ndim==2:
        psfs_over = [psfs_over]

    if not in_place:
        psfs_over = psfs_over.copy()

    # Reposition oversampled PSF to center of array using center of mass algorithm
    xyoff_psfs_over = []
    for i, psf in enumerate(psfs_over):
        xc_psf, yc_psf = get_im_cen(psf)
        xsh_sum, ysh_sum = (0, 0)
        for j in range(niter):
            if gfit:
                xc, yc = get_gfit_cen(psf, xysub=2*halfwidth+1, 
                                      return_sci=False, **kwargs)
            else:
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


###########################################################################
#    Image Cropping
###########################################################################

def crop_observation(im_full, ap, xysub, xyloc=None, delx=0, dely=0, 
                     shift_func=fourier_imshift, interp='cubic',
                     return_xy=False, fill_val=np.nan, **kwargs):
    """Crop around aperture reference location

    `xysub` specifies the desired crop size.
    if `xysub` is an array, dimension order should be [nysub,nxsub].
    Crops at pixel boundaries (no interpolation) unless delx and dely
    are specified for pixel shifting.

    `xyloc` provides a way to manually supply the central position. 
    Set `ap` to None will crop around `xyloc` or center of array.

    delx and delx will shift array by some offset before cropping
    to allow for sub-pixel shifting. To change integer crop positions,
    recommend using `xyloc` instead.

    Shift function can be fourier_imshfit, fshift, or cv_shift.
    The interp keyword only works for the latter two options.
    Consider 'lanczos' for cv_shift.

    Setting `return_xy` to True will also return the indices 
    used to perform the crop.

    Parameters
    ----------
    im_full : ndarray
        Input image.
    ap : pysiaf aperture
        Aperture to use for cropping. Will crop around the aperture
        reference point by default. Will be overridden by `xyloc`.
    xysub : int, tuple, or list
        Size of subarray to extract. If a single integer is provided,
        then a square subarray is extracted. If a tuple or list is
        provided, then it should be of the form (ny, nx).
    xyloc : tuple or list
        (x,y) pixel location around which to crop the image. If None,
        then the image aperture refernece point is used.
    
    Keyword Args
    ------------
    delx : int or float
        Pixel offset in x-direction. This shifts the image by
        some number of pixels in the x-direction. Positive values shift
        the image to the right.
    dely : int or float
        Pixel offset in y-direction. This shifts the image by
        some number of pixels in the y-direction. Positive values shift
        the image up.
    shift_func : function
        Function to use for shifting. Default is `fourier_imshift`.
        If delx and dely are both integers, then `fshift` is used.
    interp : str
        Interpolation method to use for shifting. Default is 'cubic'.
        Options are 'nearest', 'linear', 'cubic', and 'quadratic'
        for `fshift`.
    return_xy : bool
        If True, then return the x and y indices used to crop the
        image prior to any shifting from `delx` and `dely`; 
        (x1, x2, y1, y2). Default is False.
    fill_val : float
        Value to use for filling in the empty pixels after shifting.
        Default = np.nan.
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
    xy_ind = np.array([x1_init, x2_init, y1_init, y2_init])

    sh_orig = im_full.shape
    if (x2>=sh_orig[1]) or (y2>=sh_orig[0]) or (x1<0) or (y1<0):
        ny, nx = sh_orig

        # Get expansion size along x-axis
        dxp = x2 - nx + 1
        dxp = 0 if dxp<0 else dxp
        dxn = -1*x1 if x1<0 else 0
        dx = dxp + dxn

        # Get expansion size along y-axis
        dyp = y2 - ny + 1
        dyp = 0 if dyp<0 else dyp
        dyn = -1*y1 if y1<0 else 0
        dy = dyp + dyn

        # Expand image
        # TODO: This can probelmatic for some existing functions because it
        # places NaNs in the output image.
        shape_new = (2*dy+ny, 2*dx+nx)
        im_full = pad_or_cut_to_size(im_full, shape_new, fill_val=fill_val)

        xc_new, yc_new = (xc+dx, yc+dy)
        x1 = round_int(xc_new - nx_sub/2 + 0.5)
        x2 = x1 + nx_sub
        y1 = round_int(yc_new - ny_sub/2 + 0.5)
        y2 = y1 + ny_sub
    # else:
    #     xc_new, yc_new = (xc, yc)
    #     shape_new = sh_orig

    # if (x1<0) or (y1<0):
    #     dx = -1*x1 if x1<0 else 0
    #     dy = -1*y1 if y1<0 else 0

    #     # Expand image
    #     shape_new = (2*dy+shape_new[0], 2*dx+shape_new[1])
    #     im_full = pad_or_cut_to_size(im_full, shape_new)

    #     xc_new, yc_new = (xc_new+dx, yc_new+dy)
    #     x1 = round_int(xc_new - nx_sub/2 + 0.5)
    #     x2 = x1 + nx_sub
    #     y1 = round_int(yc_new - ny_sub/2 + 0.5)
    #     y2 = y1 + ny_sub

    # Perform pixel shifting
    if delx!=0 or dely!=0:
        kwargs['interp'] = interp
        # Use fshift function if only performing integer shifts
        if float(delx).is_integer() and float(dely).is_integer():
            shift_func = fshift

        # If NaNs are present, print warning and fill with zeros
        ind_nan = np.isnan(im_full)
        if np.any(ind_nan):
            # _log.warning('NaNs present in image. Filling with zeros.')
            im_full = im_full.copy()
            im_full[ind_nan] = 0

        kwargs['pad'] = True
        im_full = shift_func(im_full, delx, dely, **kwargs)
        # shift NaNs and add back in
        if np.any(ind_nan):
            ind_nan = fshift(ind_nan, delx, dely, pad=True) > 0  # Maybe >0.5?
            im_full[ind_nan] = np.nan
    
    im = im_full[y1:y2, x1:x2]
    
    if return_xy:
        return im, xy_ind
    else:
        return im


def crop_image(im, xysub, xyloc=None, **kwargs):
    """Crop input image around center using integer offsets only

    If size is exceeded, then the image is expanded and filled with NaNs.

    Parameters
    ----------
    im : ndarray
        Input image.
    xysub : int, tuple, or list
        Size of subarray to extract. If a single integer is provided,
        then a square subarray is extracted. If a tuple or list is
        provided, then it should be of the form (ny, nx).
    xyloc : tuple or list
        (x,y) pixel location around which to crop the image. If None,
        then the image center is used.
    
    Keyword Args
    ------------
    delx : int or float
        Integer pixel offset in x-direction. This shifts the image by
        some number of pixels in the x-direction. Positive values shift
        the image to the right.
    dely : int or float
        Integer pixel offset in y-direction. This shifts the image by
        some number of pixels in the y-direction. Positive values shift
        the image up.
    shift_func : function
        Function to use for shifting. Default is `fourier_imshift`.
        If delx and dely are both integers, then `fshift` is used.
    interp : str
        Interpolation method to use for shifting. Default is 'cubic'.
        Options are 'nearest', 'linear', 'cubic', and 'quadratic'
        for `fshift`.
    return_xy : bool
        If True, then return the x and y indices used to crop the
        image prior to any shifting from `delx` and `dely`; 
        (x1, x2, y1, y2). Default is False.
    fill_val : float
        Value to use for filling in the empty pixels after shifting.
        Default = np.nan.
    """
    
    return crop_observation(im, None, xysub, xyloc=xyloc, **kwargs)


def correl_images(im1, im2, mask=None):
    """ Image correlation coefficient
    
    Calculate the 2D cross-correlation coefficient between two
    images or array of images. Images must have the same x and
    y dimensions and should alredy be aligned.
    
    Parameters
    ----------
    im1 : ndarray
        Single image or image cube (nz1, ny, nx).
    im2 : ndarray
        Single image or image cube (nz2, ny, nx). 
        If both im1 and im2 are cubes, then returns
        a matrix of  coefficients.
    mask : ndarry or None
        If set, then a binary mask of 1=True and 0=False.
        Excludes pixels marked with 0s/False. Must be same
        size/shape as images (ny, nx). Any NaNs in the images
        will automatically be masked.
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

    # Mask out NaNs
    nanvals = np.sum(np.isnan(im1), axis=0) + np.sum(np.isnan(im2), axis=0)
    nan_mask = nanvals > 0
    nan_mask = nan_mask.reshape([ny1,nx1])
    if (np.sum(nan_mask) > 0) and (mask is None):
        mask = ~nan_mask
    elif (np.sum(nan_mask) > 0) and (mask is not None):
        mask = mask & ~nan_mask

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
    iymax, ixmax = np.argwhere(corr_all_fine==np.nanmax(corr_all_fine))[0]
    xsh_fine, ysh_fine = xsh_fine_vals[ixmax], ysh_fine_vals[iymax]
    
    return xsh_fine, ysh_fine


def apply_pixel_diffusion(im, pixel_sigma):
    """Apply charge diffusion kernel to image
    
    Approximates the effect of charge diffusion as a Gaussian.

    Parameters
    ----------
    im : ndarray
        Input image.
    pixel_sigma : float
        Sigma of Gaussian kernel in units of image pixels.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(im, pixel_sigma)

def gen_psf_offsets(psf, crop=65, xlim_pix=(-3,3), ylim_pix=(-3,3), dxy=0.05,
    psf_osamp=1, shift_func=fourier_imshift, ipc_vals=None, kipc=None,
    kppc=None, diffusion_sigma=None, monitor=False, prog_leave=False, **kwargs):
    """ Generate a series of downsampled cropped and shifted PSF images

    If fov_pix is odd, then crop should be odd. 
    If fov_pix is even, then crop should be even.
    
    Add IPC:
        Either ipc_vals = 0.006 or ipc_vals=[0.006,0.0004].
        The former add 0.6% to each side pixel, while the latter
        includes 0.04% to the corners. Can also supply kernel
        directly with kipc.

    Add PPC:
        Specify kppc kernel directly. This must be correctly
        oriented for the PSF image readout direction. Assumes
        single output amplifier.
    """
    
    from .image_manip import add_ipc, add_ppc

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
    if monitor:
        iter_vals = tqdm(zip(xoff_all, yoff_all), total=len(xoff_all), leave=prog_leave)
    else:
        iter_vals = zip(xoff_all, yoff_all)
    for xoff, yoff in iter_vals:
        xoff_over = xoff*psf_osamp
        yoff_over = yoff*psf_osamp
        crop_over = crop*psf_osamp

        psf_sh = crop_image(psf0, crop_over, xyloc=None, delx=-xoff_over, dely=-yoff_over,
                            shift_func=shift_func, **kwargs)
        # psf_sh = pad_or_cut_to_size(psf0, crop_over, offset_vals=(-yoff_over,-xoff_over), 
        #                             shift_func=shift_func, pad=True)

        # Apply pixel diffusion as Gaussian kernel
        if (diffusion_sigma is not None) and (diffusion_sigma > 0):
            dsig = diffusion_sigma * psf_osamp
            psf_sh = apply_pixel_diffusion(psf_sh, dsig)

        # Rebin to detector pixels
        psf_sh = frebin(psf_sh, scale=1/psf_osamp)
        psf_sh_all.append(psf_sh)

    psf_sh_all = np.asarray(psf_sh_all)
    psf_sh_all[np.isnan(psf_sh_all)] = 0
    
    # Add IPC
    if (kipc is not None) or (ipc_vals is not None):
        # Build kernel if it wasn't already specified
        if kipc is None:
            if isinstance(ipc_vals, (tuple, list, np.ndarray)):
                a1, a2 = ipc_vals
            else:
                a1, a2 = ipc_vals, 0
            kipc = np.array([[a2,a1,a2], [a1,1-4*(a1+a2),a1], [a2,a1,a2]])
        psf_sh_all = add_ipc(psf_sh_all, kernel=kipc)
    
    # Add PPC
    if (kppc is not None):
        # Build kernel if it wasn't already specified
        psf_sh_all = add_ppc(psf_sh_all, kernel=kppc, nchans=1)

    # Reshape to grid
    # sh_grid = (len(yoff_pix), len(xoff_pix))
    # xoff_all = xoff_all.reshape(sh_grid)
    # yoff_all = yoff_all.reshape(sh_grid)

    return xoff_pix, yoff_pix, psf_sh_all


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
    if is_dict and len(keys)==1:
        iter_vals = keys
    elif is_dict and len(keys)>1:
        tqdm(keys,leave=prog_leave)
    elif len(input)==1:
        iter_vals = input
    else:
        iter_vals = tqdm(input, leave=prog_leave)
    # iter_vals = tqdm(keys,leave=prog_leave) if is_dict else tqdm(input,leave=prog_leave)
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
        nanmask_psf = (psf_sh_crop==0) | np.isnan(psf_sh_crop)
        zmask2 = np.sum(nanmask_psf, axis=0) == 0
        ind_mask = rmask & zmask & zmask2 & (~bpmask)

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

def find_pix_offsets(imsub_arr, psfs, psf_osamp=1, kipc=None, kppc=None,
    diffusion_sigma=None, phase=False, bpmask_arr=None, crop=None, **kwargs):
    """Find number of pixels to offset PSFs to corrsponding images
    
    Parameters
    ----------
    imsub_arr : ndarray
        Array of cropped images
    psfs : ndarray
        Array of PSFs to align to images
    psf_osamp : int
        Oversampling factor of PSFs
    diffusion_sigma : float
        Diffusion kernel sigma value
    kipc : ndarray
        IPC kernel
    kppc : ndarray
        PPC kernel. Should already align to readout direction 
        of detector along rows.
    phase : bool
        Use phase cross-correlation to find offsets
    bpmask_arr : ndarray
        Bad pixel mask array. Should be same shape as imsub_arr.
    
    Keyword Args
    ============
    rin : float
        Exclude pixels interior to this radius.
    rout : float or None
        Exclude pixel exterior to this radius.
    xylim_pix : tuple or list
        Initial coarse step range in detector pixels.
    """

    # from webbpsf_ext.image_manip import frebin
    # from webbpsf_ext.imreg_tools import find_offsets_phase
    # from webbpsf_ext.imreg_tools import gen_psf_offsets, find_offsets2
    from .image_manip import add_ipc, add_ppc

    sh_orig = imsub_arr.shape
    if len(sh_orig)==2:
        imsub_arr = [imsub_arr]
        psfs = [psfs]
        bpmask_arr = [bpmask_arr]

    def find_pix_phase(im, psf, psf_osamp, kipc=None, kppc=None, diffusion_sigma=None, crop=15):
        # Rebin to detector sampling
        if psf_osamp!=1:
            psf = frebin(psf, scale=1/psf_osamp)

        # Add diffusion
        if (diffusion_sigma is not None) and (diffusion_sigma>0):
            psf = apply_pixel_diffusion(psf, diffusion_sigma)
        # Add IPC
        if kipc is not None:
            psf = add_ipc(psf, kernel=kipc)
        # Add PPC
        if kppc is not None:
            psf = add_ppc(psf, kernel=kppc, nchans=1)

        res = find_offsets_phase(im, psf, crop=crop, rin=0, rout=None, dxy_fine=0.01)
        return res

    def find_pix_cc(im, psf, psf_osamp, kipc=None, kppc=None, diffusion_sigma=None,
                    crop=33, bpmask=None, **kwargs):
        """Cross correlate by shifting PSF in fine steps"""

        # Create a series of coarse offset PSFs to find initial estimate
        xylim_pix = kwargs.get('xylim_pix')
        if xylim_pix is not None:
            xlim_pix = ylim_pix = xylim_pix
        else:
            xlim_pix = ylim_pix = (-10,10)
        res = gen_psf_offsets(psf, crop=crop, xlim_pix=xlim_pix, ylim_pix=xlim_pix, dxy=1,
                              psf_osamp=psf_osamp, kipc=None, kppc=None, diffusion_sigma=None,
                              prog_leave=False, shift_func=fshift, **kwargs)
        xoff_pix, yoff_pix, psf_sh_all = res
        # psf_sh_all are cropped to `crop`` value, whereas im is still input size
        xsh_coarse, ysh_coarse = find_offsets2(im, xoff_pix, yoff_pix, psf_sh_all, crop=crop,
                                               dxy_fine=0.5, bpmasks=bpmask, prog_leave=False, **kwargs)

        # Create finer grid off offset PSFs
        xlim_pix = (xsh_coarse-1.5, xsh_coarse+1.5)
        ylim_pix = (ysh_coarse-1.5, ysh_coarse+1.5)
        res = gen_psf_offsets(psf, crop=crop, xlim_pix=xlim_pix, ylim_pix=ylim_pix, dxy=0.05,
                              psf_osamp=psf_osamp, kipc=kipc, kppc=kppc, diffusion_sigma=diffusion_sigma,
                              prog_leave=False, **kwargs)

        # Perform cross correlations and interpolate at 0.01 pixel
        xoff_pix, yoff_pix, psf_sh_all = res
        return find_offsets2(im, xoff_pix, yoff_pix, psf_sh_all, crop=crop,
                             dxy_fine=0.01, bpmasks=bpmask, prog_leave=False, **kwargs)

    xysh_pix = []
    for i, im in enumerate(imsub_arr):
        if phase:
            crop = 15 if crop is None else crop
            res = find_pix_phase(im, psfs[i], psf_osamp, kipc=kipc, kppc=kppc, 
                                 diffusion_sigma=diffusion_sigma, crop=crop)
        else:
            crop = 21 if crop is None else crop
            res = find_pix_cc(im, psfs[i], psf_osamp, kipc=kipc, kppc=kppc, 
                              diffusion_sigma=diffusion_sigma, crop=crop, 
                              bpmask=bpmask_arr[i], **kwargs)
        xysh_pix.append(res)
        
    if len(sh_orig)==2:
        return xysh_pix[0]
    else:
        return np.asarray(xysh_pix)
    


###########################################################################
#    MAST and Guidestar Catalog Retrieval
###########################################################################

def download_file(filename, outdir=None, timeout=None, mast_api_token=None, 
                  overwrite=False, verbose=False):
    """ Download a MAST file
    
    Modified from M. Perrin's tools: https://github.com/mperrin/misc_jwst/blob/main/misc_jwst/guiding_analyses.py

    Parameters
    ----------
    filename : str
        Name of file to download
    outdir : str
        Output directory
    timeout : float
        Timeout in seconds to wait for download to start
    mast_api_token : str
        MAST API token
    overwrite : bool
        Overwrite existing file?
    verbose : bool
        Print extra info?
    """
    import requests, io

    from astropy.utils.console import ProgressBarOrSpinner
    from astropy.utils.data import conf
    blocksize = conf.download_block_size

    outpath = os.path.join(outdir, filename) if outdir is not None else filename

    if os.path.isfile(outpath) and (not overwrite):
        if verbose:
            print("ALREADY DOWNLOADED: ", outpath)
        return

    mast_url='https://mast.stsci.edu/api/v0.1/Download/file'
    uri_prefix = 'mast:JWST/product/'
    uri = uri_prefix + filename

    # Include MAST API token
    mast_api_token = os.environ.get('MAST_API_TOKEN') if mast_api_token is None else mast_api_token
    headers = None if mast_api_token is None else dict(Authorization=f"token {mast_api_token}")

    response = requests.get(mast_url, params=dict(uri=uri), timeout=timeout, stream=True, headers=headers)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        token1 = os.environ.get('MAST_API_TOKEN')
        token2 = os.environ.get('MAST_API_TOKEN2')
        token_list = [token1, token2]
        for i, token in enumerate(token_list):
            if (token is not None) and (token != mast_api_token):
                _log.info(f'Attempting alternate MAST_API_TOKEN...')
                headers = dict(Authorization=f"token {token}")
                response = requests.get(mast_url, params=dict(uri=uri), timeout=timeout, stream=True, headers=headers)
                try:
                    response.raise_for_status()
                except:
                    if i==len(token_list)-1:
                        raise Exception(exc)
                else:
                    break

    # Full URL of data product
    url = mast_url + uri

    if 'content-length' in response.headers:
        length = int(response.headers['content-length'])
        if length == 0:
            _log.warn(f'URL {url} has length=0')
    else:
        length = None

    # Only show progress bar if logging level is INFO or lower.
    if _log.getEffectiveLevel() <= 20:
        progress_stream = None  # Astropy default
    else:
        progress_stream = io.StringIO()

    bytes_read = 0
    msg = f'Downloading URL {url} to {outpath} ...'
    with ProgressBarOrSpinner(length, msg, file=progress_stream) as pb:
        with open(outpath, 'wb') as fd:
            for data in response.iter_content(chunk_size=blocksize):
                fd.write(data)
                bytes_read += len(data)
                if length is not None:
                    pb.update(bytes_read if bytes_read <= length else length)
                else:
                    pb.update(bytes_read)

    response.close()
    return response

def retrieve_mast_files(filenames, outdir=None, verbose=False, **kwargs):
    """Download one or more guiding data products from MAST

    Modified from M. Perrin's tools: https://github.com/mperrin/misc_jwst/blob/main/misc_jwst/guiding_analyses.py

    """

    outputs = []
    for f in filenames:
        download_file(f, outdir=outdir, **kwargs)

        # Check if files exist and append to outputs
        outfile = os.path.join(outdir, f) if outdir is not None else f
        if not os.path.isfile(outfile):
            print("ERROR: " + outfile + " failed to download.")
        else:
            if verbose:
                print("COMPLETE: ", outfile)
            outputs.append(outfile)

    return outputs

def set_params(parameters):
    """Utility function for making dicts used in MAST queries"""
    return [{"paramName":p, "values":v} for p,v in parameters.items()]


import functools
@functools.lru_cache
def find_relevant_guiding_file(sci_filename, outdir=None, verbose=False, uncals=False, **kwargs):
    """ Download fine guide file for a given science data proejct
    
    Given a filename of a JWST science file, retrieve the relevant guiding data product.
    This uses FITS keywords in the science header to determine the time period and guide mode,
    and then retrieves the file from MAST

    Modified from M. Perrin's tools: https://github.com/mperrin/misc_jwst/blob/main/misc_jwst/guiding_analyses.py


    """

    import astropy
    from astroquery.mast import Mast

    sci_hdul = fits.open(sci_filename)

    progid = sci_hdul[0].header['PROGRAM']
    obs = sci_hdul[0].header['OBSERVTN']
    guidemode = sci_hdul[0].header['PCS_MODE']

    # Set output directory if it doesn't exist
    if outdir is None:
        mast_dir = os.getenv('JWSTDOWNLOAD_OUTDIR', None)
        if mast_dir is not None:
            outdir = os.path.join(mast_dir, progid, 'fgs')
            # Create directory if it doesn't exist
            if not os.path.isdir(outdir):
                os.makedirs(outdir)

    # Set up the query
    keywords = {
        'program': [progid],
        'observtn': [obs],
        'exp_type': ['FGS_'+guidemode],
    }

    params = {
        'columns': '*',
        'filters': set_params(keywords),
    }

    # Run the web service query. This uses the specialized, lower-level webservice for the
    # guidestar queries: https://mast.stsci.edu/api/v0/_services.html#MastScienceInstrumentKeywordsGuideStar

    service = 'Mast.Jwst.Filtered.GuideStar'
    t = Mast.service_request(service, params)

    if len(t) > 0:
        # Ensure unique file names, should any be repeated over multiple observations (e.g. if parallels):
        fn = list(set(t['fileName']))
        # Set of derived Observation IDs:

        products = list(set(fn))
        # If you want the uncals
        if uncals:
            products = list(set([x.replace('_cal','_uncal') for x in fn]))
    products.sort()

    if verbose:
        print(f"For science data file: {sci_filename}")
        print("Found guiding telemetry files:")
        for p in products:
            print("   ", p)

    # Some guide files are split into multiple segments, which we have to deal with.
    guide_timestamp_parts = [fn.split('_')[2] for fn in products]
    is_segmented = ['seg' in part for part in guide_timestamp_parts]
    for i in range(len(guide_timestamp_parts)):
        if is_segmented[i]:
            guide_timestamp_parts[i] = guide_timestamp_parts[i].split('-')[0]
    guide_timestamps = np.asarray(guide_timestamp_parts, int)
    t_beg = astropy.time.Time(sci_hdul[0].header['DATE-BEG'])
    t_end = astropy.time.Time(sci_hdul[0].header['DATE-END'])
    obs_end_time = int(t_end.strftime('%Y%j%H%M%S'))

    delta_times = np.array(guide_timestamps-obs_end_time, float)
    # want to find the minimum delta which is at least positive
    # print(delta_times)
    delta_times[delta_times<0] = np.nan
    # print(delta_times)

    wmatch = np.argmin(np.abs(delta_times))
    wmatch = np.where(delta_times ==np.nanmin(delta_times))[0][0]
    delta_min = (guide_timestamps-obs_end_time)[wmatch]

    if verbose:
        print("Based on science DATE-END keyword and guiding timestamps, the matching GS file is: ")
        print("   ", products[wmatch])
        print(f"    t_end = {obs_end_time}\t delta = {delta_min}")

    if is_segmented[wmatch]:
        # We ought to fetch all the segmented GS files for that guide period
        products_to_fetch = [fn for fn in products if fn.startswith(products[wmatch][0:33])]
        if verbose:
            print("   That GS data is divided into multiple segment files:")
            print("   ".join(products_to_fetch))
    else:
        products_to_fetch = [products[wmatch],]

    outfiles = retrieve_mast_files(products_to_fetch, outdir=outdir, verbose=verbose)

    return outfiles


def get_jitter_balls(files_sci, indir, outdir=None, verbose=False, return_raw=False):
    """ Get jitter ball positions from guiding files
    
    Find the jitter ball positions from the guiding files associated with a science file.
    By default, downloads FGS fine guide files to MAST ouput directory if it exists and
    places into 'fgs' subdirectory. Otherwise, downloads to current working directory.

    Returns `(xoff_all, yoff_all)` lists of x and y positions for each science file. 
    Values are in units of arcsec relative to the first science file.

    Parameters
    ----------
    files_sci : list
        List of science file names
    indir : str
        Input directory of science files
    outdir : str
        Output directory for downloaded guiding files
    verbose : bool
        Print extra info during download?
    return_raw : bool
        Return raw xidl and yidl values instead of relative offsets?
        Default is False
    """

    from astropy.table import Table, vstack
    from astropy.time import Time

    xidl_all = []
    yidl_all = []
    for sci_filename in files_sci:
        fpath = os.path.join(indir, sci_filename)

        # Find guidestar files and read in centroid data as astropy Table
        gs_files = find_relevant_guiding_file(fpath, outdir=outdir, verbose=verbose)
        for i, gs_fn in enumerate(gs_files):
            if i==0:
                centroid_table = Table.read(gs_fn, hdu=5)
            else: 
                centroid_table = vstack([centroid_table, Table.read(gs_fn, hdu=5)], 
                                        metadata_conflicts='silent')

        # Determine start and end times for the exposure
        with fits.open(fpath) as sci_hdul:
            t_beg = Time(sci_hdul[0].header['DATE-BEG'])
            t_end = Time(sci_hdul[0].header['DATE-END'])

        # Find the subset of centroid data during exposure
        ctimes = Time(centroid_table['observatory_time'])
        mask_good = centroid_table['bad_centroid_dq_flag'] == 'GOOD'
        ctimes_during_exposure = (t_beg < ctimes ) & (ctimes < t_end) & mask_good

        xpos = centroid_table[ctimes_during_exposure]['guide_star_position_x']
        ypos = centroid_table[ctimes_during_exposure]['guide_star_position_y']

        xidl_all.append(xpos)
        yidl_all.append(ypos)

    if return_raw:
        return xidl_all, yidl_all
    else:
        # Subtract nominal position
        xmean0 = np.mean(xidl_all[0])
        ymean0 = np.mean(yidl_all[0])
        xoff_all = [(x - xmean0) for x in xidl_all]
        yoff_all = [(y - ymean0) for y in yidl_all]

        return xoff_all, yoff_all

def plot_jitter_balls(xoff_all, yoff_all, sci_filename=None, fov_size=50, 
                      save=False, save_dir=None, return_fixaxes=False):
    """ Plot jitter ball positions"""

    import matplotlib.pyplot as plt

    # Check that xoff_all and yoff_all are lists
    if not isinstance(xoff_all, list) or not isinstance(yoff_all, list):
        raise ValueError("xoff_all and yoff_all must be a list of arrays")
    
    # Convert to mas
    xoff_all = [x*1000 for x in xoff_all]
    yoff_all = [y*1000 for y in yoff_all]

    xoff_mean = np.array([np.mean(x) for x in xoff_all])
    yoff_mean = np.array([np.mean(y) for y in yoff_all])

    # Create Plots
    fig = plt.figure(figsize=(8,8), layout='constrained')

    # Create axes for scatter plot
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax.set_aspect('equal')

    # Create axes for histograms
    ax_histx = ax.inset_axes([0, 1.01, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.01, 0, 0.25, 1], sharey=ax)

    for i in range(len(xoff_all)):
        xoffsets = xoff_all[i]
        yoffsets = yoff_all[i]
        ax.scatter(xoffsets, yoffsets, alpha=0.1, marker='.', s=1)

    xylim = np.array([-1,1]) * fov_size/2
    ax.set_xlim(xylim + xoff_mean[0])
    ax.set_ylim(xylim + yoff_mean[0])
    ax.set_xlabel("FGS Centroid Offset XIDL [mas]")#, fontsize=18)
    ax.set_ylabel("FGS Centroid Offset YIDL [mas]")#, fontsize=18)

    if sci_filename is not None:
        sci_filename_act = '_'.join(os.path.basename(sci_filename).split('_')[0:2])
        fig.suptitle(f"Guiding during {sci_filename_act}_*", fontsize=14)
    else:
        fig.suptitle("Guiding during science exposures", fontsize=14)

    for i in range(len(xoff_all)):
        xc, yc = (xoff_mean[i], yoff_mean[i])
        for j, rad in enumerate([1,2,3]):
            ax.add_artist(plt.Circle( (xc, yc), rad, fill=False, color='gray', ls='--'))
            if rad<fov_size/2 and i==0:
                ax.text(j*0.5, rad+0.1, f"{rad} mas", color='gray')

    # Draw histograms
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    bsize = 0.2
    nbins = int(fov_size / bsize)
    xbins = np.linspace(xoff_mean[0]-fov_size/2, xoff_mean[0]+fov_size/2, nbins)
    ybins = np.linspace(yoff_mean[0]-fov_size/2, yoff_mean[0]+fov_size/2, nbins)
    for i in range(len(xoff_all)):
        xoffsets = xoff_all[i]
        yoffsets = yoff_all[i]
        ax_histx.hist(xoffsets, bins=xbins, alpha=0.8)
        ax_histy.hist(yoffsets, bins=ybins, orientation='horizontal', alpha=0.8)

    if save:
        figname = f'guiding_{sci_filename_act}.pdf'
        if save_dir is not None:
            figname = os.path.join(save_dir, figname)
        fig.savefig(figname, bbox_inches='tight')
        print(f"Saved {figname}")

    if return_fixaxes:
        return fig, (ax, ax_histx, ax_histy)
