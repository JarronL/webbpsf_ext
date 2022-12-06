# Import libraries
from pathlib import Path
import numpy as np
from astropy.io import fits, ascii

from webbpsf_ext.maths import jl_poly

from .utils import webbpsf, S, get_detname

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
    Read in MIRI filters from webbpsf and scale to rough peak 
    transmission throughput as indicated on JDOCS.
    """
    
    filter = filter.upper()
    filt_dir = Path(webbpsf.utils.get_webbpsf_data_path()) / 'MIRI/filters/'
    fname = f'{filter}_throughput.fits'

    bp_name = filter

    hdulist = fits.open(filt_dir / fname)
    wtemp = hdulist[1].data['WAVELENGTH']
    ttemp = hdulist[1].data['THROUGHPUT']

    # Peak values for scaling as shown on JDOX
    # TODO: Update with flight versions
    fscale_dict = {
        'F560W' : 0.30, 'F770W' : 0.35, 'F1000W': 0.35,
        'F1130W': 0.30, 'F1280W': 0.30, 'F1500W': 0.35,
        'F1800W': 0.30, 'F2100W': 0.25, 'F2550W': 0.20,

        'F1065C': 0.3, 'F1140C': 0.3, 'F1550C': 0.3, 'F2300C': 0.25,
        'FND': 0.0008
    }

    ttemp = fscale_dict[filter] * ttemp / ttemp.max()
    bp = S.ArrayBandpass(wtemp, ttemp, name=bp_name)
    hdulist.close()

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

def nircam_com_th(wave_out=None, ND_acq=False):

    # Sapphire mask transmission values for coronagraphic substrate
    fname = 'jwst_nircam_moda_com_substrate_trans.fits'
    path_com = _bp_dir / fname

    hdulist = fits.open(path_com)
    wvals = hdulist[1].data['WAVELENGTH']
    tvals = hdulist[1].data['THROUGHPUT']
    # Estimates for w<1.5um
    wvals = np.insert(wvals, 0, [0.5, 0.7, 1.2, 1.40])
    tvals = np.insert(tvals, 0, [0.2, 0.2, 0.5, 0.15])
    # Estimates for w>5.0um
    wvals = np.append(wvals, [6.00])
    tvals = np.append(tvals, [0.22])

    if ND_acq:
        ovals = nircam_com_nd(wave_out=wvals)
        tvals *= 10**(-1*ovals)

    hdulist.close()

    if wave_out is None:
        return wvals, tvals
    else:
        return np.interp(wave_out, wvals, tvals, left=0, right=0)


def nircam_com_nd(wave_out=None):
    """NIRCam COM Neutral Density squares
    
    Return optical density, where final throughput is equal
    to 10**(-1*OD).
    """
    fname = 'NDspot_ODvsWavelength.txt'
    path_ND = _bp_dir / fname
    data = ascii.read(path_ND, format='basic')

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

    if wave_out is None:
        return wdata, odata
    else:
        return np.interp(wave_out, wdata, odata, left=0, right=0)


def nircam_grism_si_ar(module, return_bp=False):
    """NIRCam grism Si and AR coating transmission data"""

    from scipy.signal import savgol_filter

    # Si plus AR coating transmission
    fname_ar = 'grism_si_2ar.txt' if module=='A' else 'grism_si_1ar.txt'
    path_ar = _bp_dir / fname_ar
    data = ascii.read(path_ar, format='csv')
    wdata = data[data.colnames[0]].data # Wavelength (um)
    tdata = data[data.colnames[1]].data / 100

    # Smooth the data, which was data-thiefed from jpeg images
    tdata = savgol_filter(tdata, 10, 3)

    if return_bp:
        bp = S.ArrayBandpass(wave=1e4*wdata, throughput=tdata)
        return bp
    else:
        return wdata, tdata

def nircam_grism_th(module, grism_order=1, wave_out=None, return_bp=False):
    """NIRCam grism throughput"""

    from scipy.interpolate import interp1d

    # Si plus AR coating transmission
    wdata, tdata = nircam_grism_si_ar(module, return_bp=False)

    # coefficients are only valid for 2.3-5.1 um
    if (module == 'A') and (grism_order==1):
        # [x^5, x^4, x^3, x^2, x^1, x^0]
        # Tom G's blaze angle of 6.16 deg
        cf_g = np.array([-0.01840, +0.36099, -2.74690, +9.95251, -16.66533, +10.27616])
        # Original PC Grate simulation (9/15/2014)
        # cf_g = np.array([+0.00307, +0.01745, -0.61644, +3.23555, -4.34493])
    elif (module == 'B') and (grism_order==1):
        # Blaze angle of 5.75 deg
        cf_g = np.array([-0.007171, +0.133052, -0.908749, +2.634983, -2.429473, -0.391573])
        # Original PC Grate simulation (9/15/2014)
        # cf_g = np.array([+0.00307, +0.01745, -0.61644, +3.23555, -4.34493])
    elif (module == 'A') and (grism_order==2):
        # TODO: Update with blaze angle of 6.16 deg
        cf_g = np.array([+0.04732, -0.77837, +4.77879, -12.97625, +13.15022])
    elif (module == 'B') and (grism_order==2):
        cf_g = np.array([+0.04732, -0.77837, +4.77879, -12.97625, +13.15022])

    # Grism blaze function
    gdata = jl_poly(wdata, cf_g[::-1])
    # Assumes Si transmission, so divide by 0.7 
    # since Si transmission accounted for in tdata
    gdata /= 0.7
    # Multiply by 0.85 for groove shadowing
    # gdata *= 0.85
    #Update from D. Jaffe suggest only 3% loss from groove shadowing (May 2022)
    gdata *= 0.97

    # Create total throughput interpolation function
    th_data = tdata*gdata
    th_func = interp1d(wdata, th_data, kind='cubic', fill_value=0)
    
    if wave_out is None:
        # Blaze function coefficients are only valid for 2.3-5.1 um
        wave_out = wdata[(wdata>=2.3) & (wdata<=5.1)]
        return_wave = True
    else:
        return_wave = False

    trans = th_func(wave_out)
    trans[trans < 0] = 0
    
    if return_bp:
        bp = S.ArrayBandpass(wave=1e4*wave_out, throughput=trans)
        return bp
    elif return_wave:
        return wdata, trans
    else:
        return trans


def qe_nircam(sca, wave=None, flight=True):
    """NIRCam QE Curves"""

    if flight:
        return qe_nircam_flight(sca, wave=wave)
    else:
        sca = get_detname(sca)
        module = sca[3]
        channel = 'LW' if sca[-1]=='5' else 'SW'
        return qe_nircam_preflight(channel, module, wave=wave)

def qe_nircam_flight(sca, wave=None):
    """NIRCam Flight QE Curve Estimates"""

    from .maths import jl_poly

    sca = get_detname(sca)
    cf_dict = {
        'NRCA1': [+0.411160074, +0.434279811, -0.092139935],
        'NRCA2': [+0.674649468, -0.226894354, +0.524256835, -0.172744162],
        'NRCA3': [+0.596092252, +0.171756196, -0.009548949],
        'NRCA4': [+0.749770825, -0.049963612, +0.057631684],
        'NRCB1': [+0.817365580, -0.110095464, +0.050932890],
        'NRCB2': [+0.426188400, +0.439977381, -0.096162371],
        'NRCB3': [+0.125705647, +0.949309791, -0.268122137],
        'NRCB4': [+0.149678825, +0.949095430, -0.281377527],
        'NRCA5': [+4.392909346, -3.765338186, +1.222109709, -0.125015154],
        'NRCB5': [+2.910495062, -2.182822116, +0.707563471, -0.071766810],
    }

    # Select coefficient
    cf = np.array(cf_dict.get(sca))

    channel = 'LW' if sca[-1]=='5' else 'SW'
    if channel=='SW':
        if wave is None:
            wave = np.arange(0.5,2.8,0.001)

        exponential = 100.
        wavecut = 2.38

        # Create smooth cut-off
        qe = jl_poly(wave, cf)
        red = wave > wavecut
        qe[red] = qe[red] * np.exp((wavecut-wave[red])*exponential)

    else:
        if wave is None:
            wave = np.arange(2.25,5.9,0.001)

        qe = jl_poly(wave, cf)
        qe[qe<0] = 0

    qe[0] = 0
    qe[-1] = 0
    bp_qe = S.ArrayBandpass(wave=1e4*wave, throughput=qe)

    return bp_qe

def qe_nircam_preflight(channel, module, wave=None):
    """ NIRCam QE

    Generate NIRCam QE curve for particular channel and module.
    These are pre-flight estimates.

    Parameters
    ==========
    channel : str
        'SW' or 'LW'
    module : str
        'A' or 'B'
    wave : ndarray
        Wavelength array in units of microns.
    """

    from .maths import jl_poly

    if channel=='SW':
        if wave is None:
            sw_wavelength = np.arange(0.5,2.8,0.001)
        else:
            sw_wavelength = wave
        sw_coeffs = np.array([0.65830,-0.05668,0.25580,-0.08350])
        sw_exponential = 100.
        sw_wavecut = 2.38

        sw_qe = jl_poly(sw_wavelength, sw_coeffs)
        red = sw_wavelength > sw_wavecut
        sw_qe[red] = sw_qe[red] * np.exp((sw_wavecut-sw_wavelength[red])*sw_exponential)

        sw_qe[0] = 0
        sw_qe[-1] = 0

        sw_qe[sw_qe<0] = 0
        bp_qe = S.ArrayBandpass(wave=1e4*sw_wavelength, throughput=sw_qe)

    else:
        if wave is None:
            lw_wavelength = np.arange(2.25,5.9,0.001)
        else:
            lw_wavelength = wave

        lw_coeffs_a = np.array([0.934871,0.051541,-0.281664,0.243867,-0.086009,0.014509,-0.001])
        lw_factor_a = 0.88
        lw_coeffs_b = np.array([2.9104951,-2.182822,0.7075635,-0.071767])

        if module.upper()=='A':
            lw_qe = lw_factor_a * jl_poly(lw_wavelength, lw_coeffs_a)
        else:
            lw_qe = jl_poly(lw_wavelength, lw_coeffs_b)

        # lw_exponential = 100.
        # lw_wavecut = 5.3
        # red = lw_wavelength > lw_wavecut
        # lw_qe[red] = lw_qe[red] * np.exp((lw_wavecut-lw_wavelength[red])*lw_exponential)

        lw_qe[0] = 0
        lw_qe[-1] = 0

        lw_qe[lw_qe<0] = 0
        bp_qe = S.ArrayBandpass(wave=1e4*lw_wavelength, throughput=lw_qe)

    return bp_qe

def qe_nirspec(wave=None):

    from scipy.interpolate import interp1d

    file = f'qe_nirspec.csv'
    file_path = str(_bp_dir / file)

    names = ['wave', 'throughput']
    data  = ascii.read(file_path, data_start=1, names=names, format='csv')

    if wave is None:
        wave = data['wave']
        th = data['throughput']
    else:
        func = interp1d(data['wave'], data['throughput'], kind='cubic', 
                        fill_value='extrapolate')
        th = func(wave)

    lw_exponential = 1
    lw_wavecut = 5.5
    red = wave > lw_wavecut
    th[red] = th[red] * np.exp((lw_wavecut-wave[red])*lw_exponential)

    th[th<0] = 0
    bp = S.ArrayBandpass(wave*1e4, th)

    return bp

def _sca_throughput_scaling(sca, filter):
    """Empirical scale factors necessary to match P330E observations"""

    sca = get_detname(sca)

    # SWA
    nrca1 = {'F070W': 1.05, 'F164N': 0.91, 'F187N': 0.89, 'F212N': 0.91}
    nrca2 = {'F070W': 1.03, 'F164N': 0.89, 'F187N': 0.89, 
             'F200W': 1.03, 'F210M': 1.03, 'F212N': 0.94}
    nrca3 = {'F164N': 0.91, 'F187N': 0.88, 'F212N': 0.90}
    nrca4 = {'F070W': 0.97, 'F115W': 1.03, 'F164N': 0.92, 'F187N': 0.89, 'F212N': 0.90}

    # SWB
    nrcb1 = {'F070W': 0.92, 'F090W': 0.92, 'F162M': 1.04, 'F164N': 0.91, 
             'F187N': 0.91, 'F200W': 1.03, 'F210M': 1.02, 'F212N': 0.89}
    nrcb2 = {'F090W': 0.98, 'F164N': 0.92, 'F187N': 0.90, 'F212N': 0.89}
    nrcb3 = {'F164N': 1.07, 'F164N': 0.91, 'F187N': 0.91, 
             'F200W': 1.02, 'F210M': 1.03, 'F212N': 0.91}
    nrcb4 = {'F070W': 1.06, 'F115W': 1.01, 'F164N': 0.92, 'F187N': 0.91, 
             'F200W': 1.03, 'F210M': 1.04, 'F212N': 0.93}

    # LWA
    nrca5 = {'F323N': 1.09, 'F360M': 0.97, 'F410M': 1.12, 'F405N': 0.94, 'F466N': 1.03}

    # LWB
    nrcb5 = {'F250M': 1.04, 'F277W': 1.05, 'F300M': 1.03, 'F335M': 1.04, 'F356W': 1.04,
             'F444W': 1.03, 'F405N': 0.94, 'F466N': 1.02, 'F480M': 0.96}

    sca_dict = {'NRCA1': nrca1, 'NRCA2': nrca2, 'NRCA3': nrca3, 'NRCA4': nrca4, 'NRCA5': nrca5,
                'NRCB1': nrcb1, 'NRCB2': nrcb2, 'NRCB3': nrcb3, 'NRCB4': nrcb4, 'NRCB5': nrcb5,}

    d = sca_dict.get(sca, None)
    if d is None:
        raise ValueError(f'SCA {sca} is not a valid NIRCam detector name!')

    scale_fact = d.get(filter, 1)
    return scale_fact


def nircam_filter(filter, pupil=None, mask=None, module=None, sca=None, ND_acq=False,
    ice_scale=0, nvr_scale=0, ote_scale=None, nc_scale=None, flight=True,
    grism_order=1, coron_substrate=False, include_blocking=True, 
    apply_scale_factors=True, **kwargs):
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
        While some filters exist in the pupil mechanisms, those should be
        specified in the `filter` argument with the option to include
        or exclude associated blocking filter via `include_blocking`
        keyword.
    mask : str, None
        Specify the coronagraphic occulter (spots or bar).
    module : str
        Module 'A' or 'B'.
    sca : str
        Valid detector name or SCA ID (NRC[A-B][1-5], 481-490). 
        This will override module if inconsitent.
    ND_acq : bool
        ND acquisition square in coronagraphic mask.
    ice_scale : float
        **Set to 0 for measured flight levels.**
        Add in additional OTE H2O absorption. This is a scale factor
        relative to 0.0131 um thickness. Also includes about 0.0150 um of
        photolyzed Carbon.
    nvr_scale : float
        **Set to 0 for measured flight levels.**
        Modify NIRCam non-volatile residue. This is a scale factor relative 
        to 0.280 um thickness already built into filter throughput curves. 
        If set to None, then assumes a scale factor of 1.0. 
        Setting ``nvr_scale=0`` will remove these contributions.
    ote_scale : float
        **Set to 0 for measured flight levels.**
        Scale factor of OTE contaminants relative to pre-flight model. 
        This is the same as setting ``ice_scale``. 
        Will override ``ice_scale`` value.
    nc_scale : float
        **Set to 0 for measured flight levels.**
        Scale factor for NIRCam contaminants relative to pre-flight model.
        This model assumes 0.189 um of NVR and 0.050 um of water ice on
        the NIRCam optical elements. Setting this keyword will remove
        NVR contributions already built into the NIRCam filter curves.
        Will override ``nvr_scale`` value.
    grism_order : int
        Option to use 2nd order grism throughputs instead. Useful if
        someone wanted to overlay the 2nd order contributions onto a 
        wide field observation.
    coron_substrate : bool
        Explicit option to include coronagraphic substrate transmission
        even if mask=None. Gives the option of using LYOT or grism pupils 
        with or without coron substrate.
    include_blocking : bool
        Include wide-band blocking filter for those filters in pupil wheel.
        These include: 'F162M', 'F164N', 'F323N', 'F405N', 'F466N', 'F470N'
    flight : bool
        Use flight bandpasses. Set to False for pre-flight vesions. Flight
        vesions include SCA-dependent QE curves.
    apply_scale_factors : bool
        Apply empirically-derived scale factors necessary to match measured
        fluxes for P330E in PID 1538.

    Returns
    -------
    :mod:`pysynphot.obsbandpass`
        A Pysynphot bandpass object.
    """
    if sca is None and module is None:
        # Set default module
        module = 'A'
    elif sca is None:
        # Module was specified, make sure it's uppercase
        module = module.upper()
    elif module is None:
        # SCA has been specified, set module
        sca = get_detname(sca)
        module = sca[3]
    else: # both module and SCA are specified
        sca = get_detname(sca)
        module = module.upper()
        # Make sure SCA and Module are consistent
        if sca[3] != module:
            new_mod = sca[3]
            _log.warning(f"Specified detector {sca} differs from Module {module}. Switching to Module {new_mod}.")
            module = new_mod

    # Choose SW or LW depending on filter
    wtemp = float(filter[1:3]) / 10
    if wtemp < 2.3:
        sca = f'NRC{module}1' if sca is None else get_detname(sca)
    else:
        sca = get_detname(sca)

    # Select filter file and read
    filter = filter.upper()
    if flight:
        filt_dir = _bp_dir / 'detector_based_throughputs'
        filt_file = f'{sca}_{filter}_system_throughput.txt'
        filt_path = str(filt_dir / filt_file)

        _log.debug(f'Reading file: {filt_file}')
        tbl = ascii.read(filt_path, format='basic')
        wave_ang, throughput = (tbl['Microns'].data * 1e4, tbl['Total_system_throughput'].data)
        bp = S.ArrayBandpass(wave_ang, throughput)
        bp_name = f"{filter}_Mod{module}"
    else:
        filt_dir = _bp_dir / 'NRC_preflight'
        filt_file = f'{filter}_nircam_plus_ote_throughput_mod{module.lower()}_sorted.txt'
        filt_path = str(filt_dir / filt_file)

        _log.debug(f'Reading file: {filt_file}')
        bp = S.FileBandpass(filt_path)
        bp_name = f"{filter}_{filt_file.split('_')[0]}"

    
    # For narrowband/mediumband filters in pupil wheel, handle blocking filter throughput
    consider_blocking = (flight and not include_blocking) or (not flight and include_blocking)
    if (filter in ['F162M', 'F164N', 'F323N', 'F405N', 'F466N', 'F470N']) and consider_blocking:
        fdir2 = _bp_dir / 'NRC_filters_only'
        if filter in ['F162M', 'F164N']:
            f2 = f'F150W2_FM.xlsx_filteronly_mod{module}_sorted.txt'
        elif filter=='F323N':
            f2 = f'F322W2_FM.xlsx_filteronly_mod{module}_sorted.txt'
        elif filter in ['F405N', 'F466N','F470N']:
            f2 = f'F444W_FM.xlsx_filteronly_mod{module}_sorted.txt'

        tbl2 = ascii.read(str(fdir2 / f2), format='basic')
        w2 = tbl2['microns'].data * 1e4
        th2 = tbl2['transmission'].data
        # Flight curves include blocking, so divide out
        if flight:
            th_new = bp.throughput / np.interp(bp.wave, w2, th2, left=0, right=0)
            th_new[np.isnan(th_new)] = 0
        else:
            th_new = bp.throughput * np.interp(bp.wave, w2, th2, left=0, right=0)
        bp_new = S.ArrayBandpass(bp.wave, th_new, name=bp.name)
        bp = bp_new        

    # Select channel (SW or LW) for minor decisions later on
    channel = 'SW' if bp.avgwave()/1e4 < 2.3 else 'LW'

    if apply_scale_factors and flight:
        if 'A5' in sca:
            # Apply a QE fix for A5 to better match P330E observations
            cf_qe_fix = np.array([1.995, -0.4525, 0.05])
            qe_fact = jl_poly(bp.wave/1e4, cf_qe_fix)
            th_new = bp.throughput * qe_fact * _sca_throughput_scaling(sca, filter)
        else:
            th_new = bp.throughput * _sca_throughput_scaling(sca, filter)
        bp = S.ArrayBandpass(bp.wave, th_new, name=bp.name)

    # Fix QE
    # if fix_lwqe and (channel=='LW'):
    #     bp_qe_orig = qe_nircam(channel, module, wave=bp.wave/1e4)
    #     bp_qe_new  = qe_nirspec(wave=bp.wave/1e4)
    #     th_new = bp.throughput
    #     indnz = bp_qe_orig.throughput>0
    #     th_new[indnz] = th_new[indnz] * bp_qe_new.throughput[indnz] / bp_qe_orig.throughput[indnz]
    #     bp = S.ArrayBandpass(bp.wave, th_new, name=bp.name)

    # Select which wavelengths to keep
    igood = bp_igood(bp, min_trans=0.005, fext=0.1)
    wgood = (bp.wave)[igood]
    w1 = wgood.min()
    w2 = wgood.max()
    wrange = w2 - w1

    # Read in grism throughput and multiply filter bandpass
    if (pupil is not None) and ('GRISM' in pupil):
        th_grism = nircam_grism_th(module, grism_order=grism_order, 
                                   wave_out=bp.wave/1e4, return_bp=False)

        # Multiply filter throughput by grism
        th_new = th_grism * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new)

        # spectral resolution in um/pixel
        # res is in pixels/um and dw is inverse
        res, dw = nircam_grism_res(pupil, module, m=grism_order)
        # Convert to Angstrom
        dw *= 10000 # Angstrom

        npts = int(wrange/dw)+1
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

        npts = int(wrange/dw)+1
        warr = np.linspace(w1, w1+dw*npts, npts)
        bp = bp.resample(warr)

    # Coronagraphic throughput modifications
    # Substrate transmission (off-axis substrate with occulting masks)
    if ((mask  is not None) and ('MASK' in mask)) or coron_substrate or ND_acq:
        # Sapphire mask transmission values for coronagraphic substrate
        # Did we explicitly set the ND acquisition square?
        # This is a special case and doesn't necessarily need to be set.
        # WebbPSF has a provision to include ND filters in the field, but we include
        # this option if the user doesn't want to figure out offset positions.
        th_coron_sub = nircam_com_th(wave_out=bp.wave/1e4, ND_acq=ND_acq)
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

            hdulist.close()

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

            hdulist.close()

        th_new = th_wedge * bp.throughput
        bp = S.ArrayBandpass(bp.wave, th_new, name=bp.name)


    # Weak Lens substrate transmission
    # TODO: Update WLP4 with newer flight version
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
        hdulist.close()

        # Throughput for WL+/-8
        hdulist = fits.open(_bp_dir / 'jwst_nircam_wlp8.fits')
        wtemp = hdulist[1].data['WAVELENGTH']
        ttemp = hdulist[1].data['THROUGHPUT']
        th_wl8 = np.interp(bp.wave/1e4, wtemp, ttemp, left=0, right=0)
        hdulist.close()

        # If two lenses
        wl48_list = ['WEAK LENS +12 (=4+8)', 'WEAK LENS -4 (=4-8)']
        if (wl_name in wl48_list):
            th_wl = th_wl4 * th_wl8
            bp_name = 'F212N' # F212N2?
            # Remove F200W contributions
            if filter=='F200W':
                th_wl /= 0.97
        elif 'WEAK LENS +4' in wl_name:
            th_wl = th_wl4
            bp_name = 'F212N' # F212N2?
            # Remove F200W contributions
            if filter=='F200W':
                th_wl /= 0.97
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
        data  = ascii.read(fname, data_start=1, names=names, format='basic')

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

    # Check [0,1] limits
    tarr[tarr<0] = 0
    tarr[tarr>1] = 1
    bp = S.ArrayBandpass(warr, tarr, name=bp_name)

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

    # Reference wavelengths in um
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

def bp_gaia(filter, release='DR2'):
    """GAIA Bandpass

    Create a bandpass class for GAIA filter to generate synthetic photometry.

    Parameters
    ==========
    filter : str
        Filters 'g', 'bp', or 'rp'
    
    """

    dir = _bp_dir / 'GAIA/'
    names = ['w_nm', 'gPb', 'gPb_Error', 'bpPb', 'bpPb_Error', 'rpPb', 'rpPb_Error']

    if release.upper()=='DR2':
        file = dir / 'GaiaDR2_RevisedPassbands.dat'
    elif release.upper()=='EDR3':
        file = dir / 'GaiaEDR3_Passbands.dat'
    else:
        raise ValueError(f"release='{release}' not recognized. Only valid releases are 'DR2' or 'EDR3'.")

    if filter.lower()=='g':
        fcol = 'gPb'
    elif filter.lower()=='bp':
        fcol = 'bpPb'
    elif filter.lower()=='rp':
        fcol = 'rpPb'
    else:
        raise ValueError(f"Filter '{filter}' not recognized. Either 'g', 'bp', or 'rp'.")

    tbl = ascii.read(file, names=names, format='basic')
    w = tbl['w_nm'] * 10  # Convert to Angstrom
    th = tbl[fcol]

    igood = th<=1
    w = w[igood]
    th = th[igood]
    th[th<0] = 0
    th[0] = 0
    th[-1] = 0

    bp = S.ArrayBandpass(w, th, name=filter)

    return bp


def filter_width(bp):
    """Return wavelength positions of filter edges at half max"""

    w, th = (bp.wave / 1e4, bp.throughput)
    wavg = bp.avgwave() / 1e4

    # Throughput at the effective wavelength
    th_mid = np.interp(wavg, w, th)
    th_half = th_mid / 2

    ind1 = (w<wavg) & (th<(th_mid+th_half) / 2)  & (th>th_half / 2)
    ind2 = (w>wavg) & (th<(th_mid+th_half) / 2)  & (th>th_half / 2)
    w_res = []
    for ind in [ind1, ind2]:
        w_arr, th_arr = (w[ind], th[ind])

        # Sort by ascending throughput values
        isort = np.argsort(th_arr)
        w_arr = w_arr[isort]
        th_arr = th_arr[isort]

        w_res.append(np.interp(th_half, th_arr, w_arr))

    return np.array(w_res)
