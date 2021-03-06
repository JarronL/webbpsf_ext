# Import libraries
from pathlib import Path
import numpy as np
from astropy.io import fits, ascii

from .utils import webbpsf, S

import logging
_log = logging.getLogger('webbpsf_ext')


from . import __path__
_bp_dir = Path(__path__[0]) / 'throughputs'

def read_filter(self, *args, **kwargs):
    if self.inst_name=='MIRI':
        return miri_filter(*args, **kwargs)
    elif self.inst_name=='NIRCam':
        return nircam_filter(*args, **kwargs)
    else:
        raise NotImplementedError(f"{self.inst_name} does not have a read bandpass function.")


def bp_igood(bp, min_trans=0.001, fext=0.05):
    """
    Given a bandpass with transmission 0.0-1.0, return the indices that
    cover only the region of interest and ignore those wavelengths with
    very low transmission less than and greater than the bandpass width.
    """
    # Select which wavelengths to use
    igood = bp.throughput > min_trans
    # Select the "good" wavelengths
    wgood = (bp.wave)[igood]
    w1 = wgood.min()
    w2 = wgood.max()
    wr = w2 - w1

    # Extend by 5% on either side
    w1 -= fext*wr
    w2 += fext*wr

    # Now choose EVERYTHING between w1 and w2 (not just th>0.001)
    ind = ((bp.wave >= w1) & (bp.wave <= w2))
    return ind

def miri_filter(filter, **kwargs):
    
    """
    No need to include pupil for the 
    """
    
    filter = filter.upper()
    filt_dir = Path(webbpsf.utils.get_webbpsf_data_path()) / 'MIRI/filters/'
    fname = f'{filter}_throughput.fits'

    bp_name = filter

    hdulist = fits.open(filt_dir / fname)
    wtemp = hdulist[1].data['WAVELENGTH']
    ttemp = hdulist[1].data['THROUGHPUT']

    fscale_dict = {
        'F560W' : 0.30, 'F770W' : 0.35, 'F1000W': 0.35,
        'F1130W': 0.30, 'F1280W': 0.30, 'F1500W': 0.35,
        'F1800W': 0.30, 'F2100W': 0.25, 'F2550W': 0.20,

        'F1065C': 0.3, 'F1140C': 0.3, 'F1550C': 0.3, 'F2300C': 0.25,
        'FND': 0.0008
    }

    ttemp = fscale_dict[filter] * ttemp / ttemp.max()
    
    bp = S.ArrayBandpass(wtemp, ttemp, name=bp_name)

    # Select which wavelengths to keep
    igood = bp_igood(bp, min_trans=0.005, fext=0.1)
    wgood = (bp.wave)[igood]
    w1 = wgood.min()
    w2 = wgood.max()
    wrange = w2 - w1

    # Resample to common dw to ensure consistency
    dw_arr = bp.wave[1:] - bp.wave[:-1]
    dw = np.median(dw_arr)
    warr = np.arange(w1,w2, dw)
    bp = bp.resample(warr)

    # Need to place zeros at either end so Pysynphot doesn't extrapolate
    warr = np.concatenate(([bp.wave.min()-dw],bp.wave,[bp.wave.max()+dw]))
    tarr = np.concatenate(([0],bp.throughput,[0]))
    bp   = S.ArrayBandpass(warr, tarr, name=bp_name)
        
    return bp


def nircam_filter(filter, pupil=None, mask=None, module=None, ND_acq=False,
    ice_scale=None, nvr_scale=None, ote_scale=None, nc_scale=None,
    grism_order=1, coron_substrate=False, **kwargs):
    """Read filter bandpass.

    Read in filter throughput curve from file generated by STScI.
    Includes: OTE, NRC mirrors, dichroic, filter curve, and detector QE.

    TODO: Account for pupil size reduction for DHS and grism observations.

    Parameters
    ----------
    filter : str
        Name of a filter.
    pupil : str, None
        NIRCam pupil elements such as grisms or lyot stops.
    mask : str, None
        Specify the coronagraphic occulter (spots or bar).
    module : str
        Module 'A' or 'B'.
    ND_acq : bool
        ND acquisition square in coronagraphic mask.
    ice_scale : float
        Add in additional OTE H2O absorption. This is a scale factor
        relative to 0.0131 um thickness. Also includes about 0.0150 um of
        photolyzed Carbon.
    nvr_scale : float
        Modify NIRCam non-volatile residue. This is a scale factor relative 
        to 0.280 um thickness already built into filter throughput curves. 
        If set to None, then assumes a scale factor of 1.0. 
        Setting nvr_scale=0 will remove these contributions.
    ote_scale : float
        Scale factor of OTE contaminants relative to End of Life model. 
        This is the same as setting ice_scale. Will override ice_scale value.
    nc_scale : float
        Scale factor for NIRCam contaminants relative to End of Life model.
        This model assumes 0.189 um of NVR and 0.050 um of water ice on
        the NIRCam optical elements. Setting this keyword will remove all
        NVR contributions built into the NIRCam filter curves.
        Overrides nvr_scale value.
    grism_order : int
        Option to use 2nd order grism throughputs instead. Useful if
        someone wanted to overlay the 2nd order contributions onto a 
        wide field observation.
    coron_substrate : bool
        Explicit option to include coronagraphic substrate transmission
        even if mask=None. Gives the option of using LYOT or grism pupils 
        with or without coron substrate.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.
    """
    if module is None: 
        module = 'A'

    # Select filter file and read
    filter = filter.upper()
    mod = module.lower()
    filt_dir = _bp_dir
    filt_file = f'{filter}_nircam_plus_ote_throughput_mod{mod}_sorted.txt'

    _log.debug('Reading file: '+filt_file)
    bp = S.FileBandpass(str(filt_dir / filt_file))
    bp_name = filter

    # Select channel (SW or LW) for minor decisions later on
    channel = 'SW' if bp.avgwave()/1e4 < 2.3 else 'LW'

    # Select which wavelengths to keep
    igood = bp_igood(bp, min_trans=0.005, fext=0.1)
    wgood = (bp.wave)[igood]
    w1 = wgood.min()
    w2 = wgood.max()
    wrange = w2 - w1

    # Read in grism throughput and multiply filter bandpass
    if (pupil is not None) and ('GRISM' in pupil):
        # Grism transmission curve follows a 3rd-order polynomial
        # The following coefficients assume that wavelength is in um
        if (module == 'A') and (grism_order==1):
            cf_g = np.array([0.068695897, -0.943894294, 4.1768413, -5.306475735])
        elif (module == 'B') and (grism_order==1):
            cf_g = np.array([0.050758635, -0.697433006, 3.086221627, -3.92089596])
        elif (module == 'A') and (grism_order==2):
            cf_g = np.array([0.05172, -0.85065, 5.22254, -14.18118, 14.37131])
        elif (module == 'B') and (grism_order==2):
            cf_g = np.array([0.03821, -0.62853, 3.85887, -10.47832, 10.61880])

        # Create polynomial function for grism throughput from coefficients
        p = np.poly1d(cf_g)
        th_grism = p(bp.wave/1e4)
        th_grism[th_grism < 0] = 0

        # Multiply filter throughput by grism
        th_new = th_grism * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)

        # spectral resolution in um/pixel
        # res is in pixels/um and dw is inverse
        res, dw = nircam_grism_res(pupil, module, m=grism_order)
        # Convert to Angstrom
        dw *= 10000 # Angstrom

        npts = np.int(wrange/dw)+1
        warr = np.linspace(w1, w1+dw*npts, npts)
        bp = bp.resample(warr)

    # Read in DHS throughput and multiply filter bandpass
    elif (pupil is not None) and ('DHS' in pupil):
        # DHS transmission curve follows a 3rd-order polynomial
        # The following coefficients assume that wavelength is in um
        cf_d = np.array([0.3192, -3.4719, 14.972, -31.979, 33.311, -12.582])
        p = np.poly1d(cf_d)
        th_dhs = p(bp.wave/1e4)
        th_dhs[th_dhs < 0] = 0
        th_dhs[bp.wave > 3e4] = 0

        # Multiply filter throughput by DHS
        th_new = th_dhs * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)

        # Mean spectral dispersion (dw/pix)
        res = 290.0
        dw = 1. / res # um/pixel
        dw *= 10000   # Angstrom/pixel

        npts = np.int(wrange/dw)+1
        warr = np.linspace(w1, w1+dw*npts, npts)
        bp = bp.resample(warr)

    # Coronagraphic throughput modifications
    # Substrate transmission (off-axis substrate with occulting masks)
    if ((mask  is not None) and ('MASK' in mask)) or coron_substrate or ND_acq:
        # Sapphire mask transmission values for coronagraphic substrate
        hdulist = fits.open(_bp_dir / 'jwst_nircam_moda_com_substrate_trans.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']
        # Estimates for w<1.5um
        wtemp = np.insert(wtemp, 0, [0.5, 0.7, 1.2, 1.40])
        ttemp = np.insert(ttemp, 0, [0.2, 0.2, 0.5, 0.15])
        # Estimates for w>5.0um
        wtemp = np.append(wtemp, [6.00])
        ttemp = np.append(ttemp, [0.22])

        # Did we explicitly set the ND acquisition square?
        # This is a special case and doesn't necessarily need to be set.
        # WebbPSF has a provision to include ND filters in the field, but we include
        # this option if the user doesn't want to figure out offset positions.
        if ND_acq:
            fname = 'NDspot_ODvsWavelength.txt'
            path_ND = _bp_dir / fname
            data = ascii.read(path_ND)

            wdata = data[data.colnames[0]].data # Wavelength (um)
            odata = data[data.colnames[1]].data # Optical Density
            # Estimates for w<1.5um
            wdata = np.insert(wdata, 0, [0.5])
            odata = np.insert(odata, 0, [3.8])
            # Estimates for w>5.0um
            wdata = np.append(wdata, [6.00])
            odata = np.append(odata, [2.97])

            # CV3 data suggests OD needs to be multiplied by 0.93
            # compared to Barr measurements
            odata *= 0.93

            otemp = np.interp(wtemp, wdata, odata, left=0, right=0)
            ttemp *= 10**(-1*otemp)

        # Interpolate substrate transmission onto filter wavelength grid and multiply
        th_coron_sub = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)
        th_new = th_coron_sub * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)


    # Lyot stop wedge modifications 
    # Substrate transmission (located in pupil wheel to deflect beam)
    if (pupil is not None) and ('LYOT' in pupil):

        # Transmission values for wedges in Lyot stop
        if 'SW' in channel:
            fname = 'jwst_nircam_sw-lyot_trans_modmean.fits'
            hdulist = fits.open(_bp_dir / fname)
            wtemp = hdulist[1].data['WAVELENGTH']
            ttemp = hdulist[1].data['THROUGHPUT']
            # Estimates for w<1.5um
            wtemp = np.insert(wtemp, 0, [0.50, 1.00])
            ttemp = np.insert(ttemp, 0, [0.95, 0.95])
            # Estimates for w>2.3um
            wtemp = np.append(wtemp, [2.50,3.00])
            ttemp = np.append(ttemp, [0.85,0.85])
            # Interpolate substrate transmission onto filter wavelength grid
            th_wedge = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        elif 'LW' in channel:
            fname = 'jwst_nircam_lw-lyot_trans_modmean.fits'
            hdulist = fits.open(_bp_dir / fname)
            wtemp = hdulist[1].data['WAVELENGTH']
            ttemp = hdulist[1].data['THROUGHPUT']
            ttemp *= 100 # Factors of 100 error in saved values

            # Smooth the raw data
            ws = 200
            s = np.r_[ttemp[ws-1:0:-1],ttemp,ttemp[-1:-ws:-1]]
            w = np.blackman(ws)
            y = np.convolve(w/w.sum(),s,mode='valid')
            ttemp = y[int((ws/2-1)):int(-(ws/2))]

            # Estimates for w<2.3um
            wtemp = np.insert(wtemp, 0, [1.00])
            ttemp = np.insert(ttemp, 0, [0.95])
            # Estimates for w>5.0um
            wtemp = np.append(wtemp, [6.0])
            ttemp = np.append(ttemp, [0.9])
            # Interpolate substrate transmission onto filter wavelength grid
            th_wedge = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        th_new = th_wedge * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new, name=bp.name)


    # Weak Lens substrate transmission
    if (pupil is not None) and (('WL' in pupil) or ('WEAK LENS' in pupil)):

        if 'WL' in pupil:
            wl_alt = {'WLP4' :'WEAK LENS +4', 
                      'WLP8' :'WEAK LENS +8', 
                      'WLP12':'WEAK LENS +12 (=4+8)', 
                      'WLM4' :'WEAK LENS -4 (=4-8)',
                      'WLM8' :'WEAK LENS -8'}
            wl_name = wl_alt.get(pupil, pupil)
        else:
            wl_name = pupil

        # Throughput for WL+4
        hdulist = fits.open(_bp_dir / 'jwst_nircam_wlp4.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']
        th_wl4 = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        # Throughput for WL+/-8
        hdulist = fits.open(_bp_dir / 'jwst_nircam_wlp8.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']
        th_wl8 = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)

        # If two lenses
        wl48_list = ['WEAK LENS +12 (=4+8)', 'WEAK LENS -4 (=4-8)']
        if (wl_name in wl48_list):
            th_wl = th_wl4 * th_wl8
            bp_name = 'F212N'
        elif 'WEAK LENS +4' in wl_name:
            th_wl = th_wl4
            bp_name = 'F212N'
        else:
            th_wl = th_wl8
            
        th_new = th_wl * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)

        # Select which wavelengths to keep
        igood = bp_igood(bp, min_trans=0.005, fext=0.1)
        wgood = (bp.wave)[igood]
        w1 = wgood.min()
        w2 = wgood.max()
        wrange = w2 - w1

    # OTE scaling (use ice_scale keyword)
    if ote_scale is not None:
        ice_scale = ote_scale
    if nc_scale is not None:
        nvr_scale = 0
    # Water ice and NVR additions (for LW channel only)
    if ((ice_scale is not None) or (nvr_scale is not None)) and ('LW' in channel):
        fname = _bp_dir / 'ote_nc_sim_1.00.txt'
        names = ['Wave', 't_ice', 't_nvr', 't_sys']
        data  = ascii.read(fname, data_start=1, names=names)

        wtemp = data['Wave']
        wtemp = np.insert(wtemp, 0, [1.0]) # Estimates for w<2.5um
        wtemp = np.append(wtemp, [6.0])    # Estimates for w>5.0um

        th_new = bp.throughput
        if ice_scale is not None:
            ttemp = data['t_ice']
            ttemp = np.insert(ttemp, 0, [1.0]) # Estimates for w<2.5um
            ttemp = np.append(ttemp, [1.0])    # Estimates for w>5.0um
            # Interpolate transmission onto filter wavelength grid
            ttemp = np.interp(bp.wave/1e4, wtemp, ttemp)#, left=0, right=0)
            
            # Scale is fraction of absorption feature depth, not of layer thickness
            th_new = th_new * (1 - ice_scale * (1 - ttemp))
            # th_ice = np.exp(ice_scale * np.log(ttemp))
            # th_new = th_ice * th_new

        if nvr_scale is not None:
            ttemp = data['t_nvr']
            ttemp = np.insert(ttemp, 0, [1.0]) # Estimates for w<2.5um
            ttemp = np.append(ttemp, [1.0])    # Estimates for w>5.0um
            # Interpolate transmission onto filter wavelength grid
            ttemp = np.interp(bp.wave/1e4, wtemp, ttemp)#, left=0, right=0)
            
            # Scale is fraction of absorption feature depth, not of layer thickness
            # First, remove NVR contributions already included in throughput curve
            th_new = th_new / ttemp
            th_new = th_new * (1 - nvr_scale * (1 - ttemp))
            
            # The "-1" removes NVR contributions already included in
            # NIRCam throughput curves
            # th_nvr = np.exp((nvr_scale-1) * np.log(ttemp))
            # th_new = th_nvr * th_new
            
        if nc_scale is not None:
            names = ['Wave', 'coeff'] # coeff is per um path length
            data_ice  = ascii.read(_bp_dir / 'h2o_abs.txt', names=names)
            data_nvr  = ascii.read(_bp_dir / 'nvr_abs.txt', names=names)
    
            w_ice = data_ice['Wave']
            a_ice = data_ice['coeff']
            a_ice = np.interp(bp.wave/1e4, w_ice, a_ice)

            w_nvr = data_nvr['Wave']
            a_nvr = data_nvr['coeff']
            a_nvr = np.interp(bp.wave/1e4, w_nvr, a_nvr)

            ttemp = np.exp(-0.189 * a_nvr - 0.050 * a_ice)
            th_new = th_new * (1 - nc_scale * (1 - ttemp))
            
            # ttemp = np.exp(-nc_scale*(a_nvr*0.189 + a_ice*0.05))
            # th_new = ttemp * th_new

        # Create new bandpass
        bp = S.ArrayBandpass(bp.wave, th_new)


    # Resample to common dw to ensure consistency
    dw_arr = bp.wave[1:] - bp.wave[:-1]
    #if not np.isclose(dw_arr.min(),dw_arr.max()):
    dw = np.median(dw_arr)
    warr = np.arange(w1,w2, dw)
    bp = bp.resample(warr)

    # Need to place zeros at either end so Pysynphot doesn't extrapolate
    warr = np.concatenate(([bp.wave.min()-dw],bp.wave,[bp.wave.max()+dw]))
    tarr = np.concatenate(([0],bp.throughput,[0]))
    bp   = S.ArrayBandpass(warr, tarr, name=bp_name)

    return bp

def nircam_grism_res(pupil='GRISM', module='A', m=1):
    """NIRCam Grism resolution

    Based on the pupil input and module, return the spectral
    dispersion and resolution as a tuple (res, dw).

    Parameters
    ----------
    pupil : str
        'GRISMC' or 'GRISMR', otherwise assume res=1000 pix/um.
        'GRISM0' is GRISMR; 'GRISM90' is GRISMC
    module : str
        'A' or 'B'
    m : int
        Spectral order (1 or 2).
    """

    # Option for GRISM0/GRISM90
    if 'GRISM0' in pupil:
        pupil = 'GRISMR'
    elif 'GRISM90' in pupil:
        pupil = 'GRISMC'

    # Mean spectral dispersion in number of pixels per um
    if ('GRISMC' in pupil) and (module == 'A'):
        res = 1003.12
    elif ('GRISMR' in pupil)  and (module == 'A'):
        res = 996.48
    elif ('GRISMC' in pupil) and (module == 'B'):
        res = 1008.64
    elif ('GRISMR' in pupil)  and (module == 'B'):
        res = 1009.13
    else:
        res = 1000.0

    if m==2:
        res *= 2

    # Spectral resolution in um/pixel
    dw = 1. / res

    return (res, dw)

def nircam_grism_wref(pupil='GRISM', module='A'):
    """NIRCam Grism undeviated wavelength"""

    # Option for GRISM0/GRISM90
    if 'GRISM0' in pupil:
        pupil = 'GRISMR'
    elif 'GRISM90' in pupil:
        pupil = 'GRISMC'

    # Mean spectral dispersion in number of pixels per um
    if ('GRISMC' in pupil) and (module == 'A'):
        wref = 3.978
    elif ('GRISMR' in pupil)  and (module == 'A'):
        wref = 3.937
    elif ('GRISMC' in pupil) and (module == 'B'):
        wref = 3.923
    elif ('GRISMR' in pupil)  and (module == 'B'):
        wref = 3.960
    else:
        wref = 3.95

    return wref


def niriss_grism_res(m=1):
    """Grism resolution

    Based on the pupil input and module, return the spectral
    dispersion and resolution as a tuple (res, dw).

    Parameters
    ----------
    m : int
        Spectral order (1 or 2).
    """

    # Spectral resolution in um/pixel
    dw = 0.00478
    res = 1. / dw

    if m==2:
        res *= 2
        dw *= 0.5

    return (res, dw)

def niriss_grism_wref():
    """NIRISS Grism undeviated wavelength (um)"""
    return 1.0


def bp_2mass(filter):
    """2MASS Bandpass

    Create a 2MASS J, H, or Ks filter bandpass used to generate
    synthetic photometry.

    Parameters
    ----------
    filter : str
        Filter 'j', 'h', or 'k'.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.

    """

    dir = _bp_dir / '2MASS/'
    if 'j' in filter.lower():
        file = '2mass_j.txt'
        name = 'J-Band'
    elif 'h' in filter.lower():
        file = '2mass_h.txt'
        name = 'H-Band'
    elif 'k' in filter.lower():
        file = '2mass_ks.txt'
        name = 'Ks-Band'
    else:
        raise ValueError('{} not a valid 2MASS filter'.format(filter))

    tbl = ascii.read(dir / file, names=['Wave', 'Throughput'])
    bp = S.ArrayBandpass(tbl['Wave']*1e4, tbl['Throughput'], name=name)

    return bp

def bp_wise(filter):
    """WISE Bandpass

    Create a WISE W1-W4 filter bandpass used to generate
    synthetic photometry.

    Parameters
    ----------
    filter : str
        Filter 'w1', 'w2', 'w3', or 'w4'.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.

    """

    dir = _bp_dir / 'WISE/'
    if 'w1' in filter.lower():
        file = 'RSR-W1.txt'
        name = 'W1'
    elif 'w2' in filter.lower():
        file = 'RSR-W2.txt'
        name = 'W2'
    elif 'w3' in filter.lower():
        file = 'RSR-W3.txt'
        name = 'W3'
    elif 'w4' in filter.lower():
        file = 'RSR-W4.txt'
        name = 'W4'
    else:
        raise ValueError('{} not a valid WISE filter'.format(filter))

    tbl = ascii.read(dir / file, data_start=0)
    bp = S.ArrayBandpass(tbl['col1']*1e4, tbl['col2'], name=name)

    return bp
