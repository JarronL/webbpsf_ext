# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import os

import scipy
# from scipy.sparse.construct import random
import scipy.stats
from scipy.stats import arcsine
from scipy.interpolate import interp1d

from astropy.io import fits
import astropy.units as u

# WebbPSF
import webbpsf
from webbpsf.opds import OTE_Linear_Model_WSS
from webbpsf.utils import get_webbpsf_data_path

# Logging
from . import conf
from .logging_utils import setup_logging
import logging
_log = logging.getLogger('webbpsf_ext')

# Progress bar
from tqdm.auto import trange, tqdm

__epsilon = np.finfo(float).eps

def OPDFile_to_HDUList(file, slice=0):
    """
    Make a picklable HDUList for ingesting into multiproccessor WebbPSF
    helper function.
    """

    try:
        hdul = fits.open(file)
    except FileNotFoundError:
        # Check WebbPSF instrument OPD directory
        if 'NIRCam' in file:
            inst = 'NIRCam'
        elif 'MIRI' in file:
            inst = 'MIRI'
        elif 'NIRSpec' in file:
            inst = 'NIRSpec'
        elif 'NIRISS' in file:
            inst = 'NIRISS'
        elif 'FGS' in file:
            inst = 'FGS'

        if 'JWST_OTE_OPD' in file:
            # Location of JWST_OTE_OPD_RevAA_prelaunch_predicted.fits.gz
            opd_dir = get_webbpsf_data_path()
        else:
            opd_dir = os.path.join(get_webbpsf_data_path(),inst,'OPD')
        hdul = fits.open(os.path.join(opd_dir, file))
    ndim = len(hdul[0].data.shape)

    if ndim==3:
        opd_im = hdul[0].data[slice,:,:]
    else:
        opd_im = hdul[0].data

    hdu_new = fits.PrimaryHDU(opd_im)
    hdu_new.header = hdul[0].header.copy()
    opd_hdul = fits.HDUList([hdu_new])

    hdul.close()

    return opd_hdul


class OTE_WFE_Drift_Model(OTE_Linear_Model_WSS):
    """
    OPD subclass for calculating OPD drift values over time.
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        opdfile : str or fits.HDUList
            FITS file to load an OPD from. The OPD data must be specified in microns.
        opd_index : int, optional
            Index of a datacube to load OPD from, if the selected 
            extension contains a datacube.
        transmission : str or None
            FITS file for pupil mask, with throughput from 0-1. 
            If not explicitly provided, will be inferred from 
            wherever is nonzero in the OPD file.
        segment_mask_file : str
            FITS file for pupil mask, with throughput from 0-1. If not explicitly provided, will
            use JWpupil_segments_RevW_npix1024.fits, or equivalent for other value of npix
        zero : bool
            Set an OPD to precisely zero everywhere.
        rm_ptt : bool
            Remove piston, tip, and tilt? This is mostly for visualizing 
            the higher order parts of the LOM. Default: False.
        v2v3 : tuple of 2 astropy.Quantities
            Tuple giving V2,v3 coordinates as quantities, typically in arcminutes, or None to default to
            the master chief ray location between the two NIRCam modules.
        include_nominal_field_dependence : bool
            Include the Zernike polynomial model for OTE field dependence for the nominal OTE.
            Note, if OPD is None, then this will be ignored and the nominal field dependence will be disabled.
        control_point_fieldpoint: str
            A parameter used in the field dependence model for a misaligned secondary mirror.
            Name of the field point where the OTE MIMF control point is located, on instrument defined by "control_point_instr".
            Default: 'nrca3_full'.
            The OTE control point is the field point to which the OTE has been aligned and defines the field angles
            for the field-dependent SM pose aberrations.
        npix : int
            Size of OPD: npix x npix
        """
        
        # Initialize OTE_Linear_Model_WSS
        OTE_Linear_Model_WSS.__init__(self, **kwargs)
        
        # Initialize delta OPD normalized images
        self.dopd_thermal = None
        self.dopd_frill   = None
        self.dopd_iec     = None
        
        # Initialize normalized delta OPD images
        log_prev = conf.logging_level
        if 'WARN' not in log_prev:
            setup_logging('WARN', verbose=False)
        self._calc_delta_opds()
        if 'WARN' not in log_prev:
            setup_logging(log_prev, verbose=False)
        
    def reset(self, verbose=True):
        """ Reset an OPD to the state it was loaded from disk.

        i.e. undo all segment moves.
        """
        self._frill_wfe_amplitude = 0
        self._iec_wfe_amplitude = 0
        self.opd = self._opd_original.copy()
        self.segment_state *= 0
        if verbose:
            _log.info("Reset to unperturbed OPD")
        
        
    def _calc_delta_opds(self, thermal=True, frill=True, iec=True):
        """
        Calculate delta OPDs for the three components and save to
        class properties. Each delta OPD image will be normalized
        such that the nm RMS WFE is equal to 1. 
        
        """
        
        # Set everything to initial state
        self.reset(verbose=False)
        
        # Calculate thermal dOPD
        if thermal:
            self.thermal_slew(1*u.day)
            # self.opd has now been updated to drifted OPD
            # Calculate delta OPD and save into self.opd attribute
            # This is because self.rms() uses the image in self.opd
            self.opd -= self._opd_original
            # scale by RMS of delta OPD, and save
            self.dopd_thermal = self.opd / self.rms() 

        # Calculate frill dOPD
        if frill:
            # Explicitly set thermal component to 0
            self.thermal_slew(0*u.min, scaling=0, delay_update=True)
            self.apply_frill_drift(amplitude=1)
            # self.opd has now been updated to drifted OPD
            # Temporarily calculate delta and calc rms
            self.opd -= self._opd_original
            # scale by RMS of delta OPD, and save
            self.dopd_frill = self.opd / self.rms() 

        # Calculate IEC dOPD
        if iec:
            # Explicitly set thermal and frill components to 0
            self.thermal_slew(0*u.min, scaling=0, delay_update=True)
            self.apply_frill_drift(amplitude=0, delay_update=True)
            self.apply_iec_drift(amplitude=1)
            # self.opd has now been updated to drifted OPD
            # Temporarily calculate delta and calc rms
            self.opd -= self._opd_original
            # scale by RMS of delta OPD, and save
            self.dopd_iec = self.opd / self.rms() 
        
        # Back to initial state
        self.reset(verbose=False)

        
    def calc_rms(self, arr, segname=None):
        """Calculate RMS of input images"""

        # RMS for a single image
        def rms_im(im):
            """ Find RMS of an image by excluding pixels with 0s, NaNs, or Infs"""
            ind = (im != 0) & (np.isfinite(im))
            res = 0 if len(im[ind]) == 0 else np.sqrt(np.mean(im[ind] ** 2))
            res = 0 if np.isnan(res) else res
            return res
        
        # Reshape into a 3-dimension cube for consistency
        if len(arr.shape) == 3:
            nz,ny,nx = arr.shape
        else:
            ny,nx = arr.shape
            nz = 1
            arr = arr.reshape([nz,ny,nx])

        if segname is None:
            # RMS of whole aperture
            rms = np.asarray([rms_im(im) for im in arr])
        else:
            # RMS of specified segment
            assert (segname in self.segnames)
            iseg = np.where(self.segnames == segname)[0][0] + 1  # segment index from 1 - 18
            seg_mask = self._segment_masks == iseg
            arr_seg = arr[:,seg_mask]
            rms = np.asarray([rms_im(im) for im in arr_seg])

        # If single image, remove first dimension
        if nz==1:
            rms = rms[0]

        return rms
        
    def slew_scaling(self, start_angle, end_angle):
        """ WFE scaling due to slew angle
        
        Scale the WSS Hexike components based on slew pitch angles.
        
        Parameters
        ----------
        start_angle : float
            The starting sun pitch angle, in degrees between -5 and +45
        end_angle : float
            The ending sun pitch angle, in degrees between -5 and +45
        """
        num = np.sin(np.radians(end_angle)) - np.sin(np.radians(start_angle))
        den = np.sin(np.radians(45.)) - np.sin(np.radians(-5.))

        return num / den
    
    def gen_frill_drift(self, delta_time, start_angle=-5, end_angle=45, case='BOL'):
        """ Frill WFE drift scaling
        
        Function to determine the factor to scale the delta OPD associated with
        frill tensioning. Returns the RMS WFE (nm) depending on time and slew
        angles.
        
        Parameters
        ----------
        delta_time : astropy.units quantity object
            The time since a slew occurred.
        start_angle : float
            The starting sun pitch angle, in degrees between -5 and +45
        end_angle : float
            The ending sun pitch angle, in degrees between -5 and +45
        case : string
            either "BOL" for current best estimate at beginning of life, or
            "EOL" for more conservative prediction at end of life. The amplitude
            of the frill drift is roughly 2x lower for BOL (8.6 nm after 2 days)
            versus EOL (18.4 nm after 2 days).

        """

        frill_hours = np.array([
            0.00, 0.55, 1.00, 1.60, 2.23, 2.85, 3.47, 4.09, 
            4.71, 5.33, 5.94, 6.56, 7.78, 9.00, 9.60, 11.41, 
            12.92, 15.02, 18.00, 21.57, 23.94, 26.90, 32.22, 
            35.76, 41.07, 45.20, 50.50, 100.58
        ])
        # Normalized frill drift amplitude
        frill_wfe_drift_norm = np.array([
            0.000, 0.069, 0.120, 0.176, 0.232, 0.277,
            0.320, 0.362, 0.404, 0.444, 0.480, 0.514,
            0.570, 0.623, 0.648, 0.709, 0.758, 0.807,
            0.862, 0.906, 0.930, 0.948, 0.972, 0.981,
            0.991, 0.995, 0.998, 1.000
        ])

        # Create interpolation function
        finterp = interp1d(frill_hours, frill_wfe_drift_norm,
                           kind='cubic', fill_value=(0, 1), bounds_error=False)
                                             
        # Convert input time to hours and get normalized amplitude
        time_hour = delta_time.to(u.hour).value
        amp_norm = finterp(time_hour)
        
        # Scale height from either EOL or BOL (nm RMS)
        # Assuming slew angles from -5 to +45 deg
        if case=='EOL':
            wfe_drift_rms = 18.4 * amp_norm
        elif case=='BOL':
            wfe_drift_rms = 8.6 * amp_norm
        else:
            print(f'case={case} is not recognized')

        # Get scale factor based on start and end angle solar elongation angles
        scaling = self.slew_scaling(start_angle, end_angle)
        wfe_drift_rms *= scaling 

        return wfe_drift_rms
    
    
    def gen_thermal_drift(self, delta_time, start_angle=-5, end_angle=45, case='BOL'):
        """ Thermal WFE drift scaling
        
        Function to determine the factor to scale the delta OPD associated with
        OTE backplane thermal distortion. Returns the RMS WFE (nm) depending on 
        time and slew angles.
        
        Parameters
        ----------
        delta_time : astropy.units quantity object
            The time since a slew occurred.
        start_angle : float
            The starting sun pitch angle, in degrees between -5 and +45
        end_angle : float
            The ending sun pitch angle, in degrees between -5 and +45
        case : string
            either "BOL" for current best estimate at beginning of life, or
            "EOL" for more conservative prediction at end of life. The amplitude
            of the frill drift is roughly 3x lower for BOL (13 nm after 14 days)
            versus EOL (43 nm after 14 days).

        """
        
        thermal_hours = np.array([
            0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,
            11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21.,
            22.,  23.,  24.,  48.,  72.,  96., 120., 144., 168., 192., 216.,
            240., 264., 288., 312., 336., 360., 384., 408., 432., 456., 480., 800.
        ])
        
        thermal_wfe_drift_norm = np.array([
            0.0000, 0.0134, 0.0259, 0.0375, 0.0484, 0.0587, 0.0685, 0.0777, 0.0865, 
            0.0950, 0.1031, 0.1109, 0.1185, 0.1259, 0.1330, 0.1400, 0.1468, 0.1534, 
            0.1600, 0.1664, 0.1727, 0.1789, 0.1850, 0.1910, 0.1970, 0.3243, 0.4315, 
            0.5227, 0.5999, 0.6650, 0.7197, 0.7655, 0.8038, 0.8358, 0.8625, 0.8849, 
            0.9035, 0.9191, 0.9322, 0.9431, 0.9522, 0.9598, 0.9662, 0.9716, 1.0000
        ])
        
        # Create interpolation function
        finterp = interp1d(thermal_hours, thermal_wfe_drift_norm,
                           kind='cubic', fill_value=(0, 1), bounds_error=False)
                                             
        # Convert input time to hours and get normalized amplitude
        time_hour = delta_time.to(u.hour).value
        amp_norm = finterp(time_hour)
        
        # Normalize to 14 days (336 hours)
        amp_norm /= finterp(336)
        
        # Scale height from either EOL or BOL (nm RMS)
        # Assuming full slew angle from -5 to +45 deg
        if case=='EOL':
            wfe_drift_rms = 45.0 * amp_norm
        elif case=='BOL':
            wfe_drift_rms = 13.0 * amp_norm
        else:
            print(f'case={case} is not recognized')

        # Get scale factor based on start and end angle solar elongation angles
        scaling = self.slew_scaling(start_angle, end_angle)
        wfe_drift_rms *= scaling 

        return wfe_drift_rms

    
    def gen_iec_series(self, delta_time, amplitude=3.5, period=5.0, 
        interp_kind='linear', random_seed=None):
        """Create a series of IEC WFE scale factors
        
        Create a series of random IEC heater state changes based on 
        arcsine distribution. 
        
        Parameters
        ----------
        delta_time : astropy.units quantity object array
            Time series of atropy units to interpolate IEC amplitudes
        
        Keyword Args
        ------------
        amplitude : float
            Full amplitude of arcsine distribution. Values will range
            from -0.5*amplitude to +0.5*amplitude.
        period : float
            Period in minutes of IEC oscillations. Usually 3-5 minutes.
        random_seed : int
            Provide a random seed value between 0 and (2**32)-1 to generate
            reproducible random values.
        interp_kind : str or int
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.
            Default is 'linear'.
        """
        
        # Convert time array to minutes and get values
        if isinstance(delta_time, (u.Quantity)):
            time_arr_minutes = np.array(delta_time.to(u.min).value)
        else:
            time_arr_minutes = delta_time
        
        # Create a series of random IEC heater state changes based on arcsin distribution
        dt = period
        nsamp = int(np.max(time_arr_minutes)/dt) + 2
        tvals = np.arange(nsamp) * dt

        # Random values between 0 and 1
        arcsine_rand = arcsine.rvs(size=nsamp, random_state=random_seed)
        # Scale by amplitude
        wfe_iec_all = arcsine_rand * amplitude - amplitude / 2

        # res = np.interp(time_arr_minutes, tvals, wfe_iec_all)

        finterp = interp1d(tvals, wfe_iec_all, kind=interp_kind,
                           fill_value=0, bounds_error=False)
        res = finterp(time_arr_minutes)

        return res
    
    
    def gen_delta_opds(self, delta_time, start_angle=-5, end_angle=45, 
                       do_thermal=True, do_frill=True, do_iec=True, 
                       case='BOL', return_wfe_amps=True, return_dopd_fin=True,
                       random_seed=None, **kwargs):
        
        """Create series of delta OPDs
        
        Generate a series of delta OPDS, the result of which is
        a combination of thermal, frill, and IEC effects. The
        thermal and frill values are dependent on time, start/end
        slew angles, and case ('BOL' or 'EOL'). Delta OPD contributions
        from the IEC heater switching are treated as random state
        switches assuming an arcsine distribution.
        
        Parameters
        ----------
        delta_time : astropy.units quantity object
            An array of times assuming astropy units.
        start_angle : float
            The starting sun pitch angle, in degrees between -5 and +45.
        end_angle : float
            The ending sun pitch angle, in degrees between -5 and +45.
        case : string
            Either "BOL" for current best estimate at beginning of life, or
            "EOL" for more conservative prediction at end of life.
        do_thermal : bool
            Include thermal slew component? Mostly for debugging purposes.
        do_frill : bool
            Include frill component? Mostly for debugging purposes.
        do_iec : bool
            Include IEC component? Good to exclude if calling this function
            repeatedly for evolution of multiple slews, then add IEC later.
        return_wfe_amps : bool
            Return a dictionary that provides the RMS WFE (nm) of each
            component at each time step.
        return_dopd_fin : bool
            Option to exclude calculating final delta OPD in case we only
            want the final RMS WFE dictionary.
        random_seed : int
            Random seed to pass to IEC generation.
        """
        
        if (not return_wfe_amps) and (not return_dopd_fin):
            _log.warning('Must specify `return_wfe_amps` and/or `return_dopd_fin`')
            return
        
        try:
            nz = len(delta_time)
        except TypeError:
            nz = 1
        ny,nx = self.opd.shape
        
        # Thermal drift amplitudes
        if do_thermal:        
            amp_thermal = self.gen_thermal_drift(delta_time, case=case,
                                                 start_angle=start_angle, 
                                                 end_angle=end_angle)
        else:
            amp_thermal = np.zeros(nz) if nz>1 else 0

        # Frill drift amplitudes
        if do_frill:
            amp_frill = self.gen_frill_drift(delta_time, case=case,
                                             start_angle=start_angle, 
                                             end_angle=end_angle)
        else:
            amp_frill = np.zeros(nz) if nz>1 else 0
        
        # Random IEC amplitudes
        if do_iec:
            amp_iec = self.gen_iec_series(delta_time, random_seed=random_seed, **kwargs)
            if nz>1:
                amp_iec[0] = 0
        else:
            amp_iec = np.zeros(nz) if nz>1 else 0
        
        
        # Add OPD deltas
        if return_dopd_fin:
            delta_opd_fin = np.zeros([nz,ny,nx])
            if do_thermal:
                amp = np.reshape(amp_thermal, [-1,1,1])
                delta_opd_fin += self.dopd_thermal.reshape([1,ny,nx]) * amp
            if do_frill:
                amp = np.reshape(amp_frill, [-1,1,1])
                delta_opd_fin += self.dopd_frill.reshape([1,ny,nx]) * amp
            if do_iec:
                amp = np.reshape(amp_iec, [-1,1,1])
                delta_opd_fin += self.dopd_iec.reshape([1,ny,nx]) * amp
                
            if nz==1:
                delta_opd_fin = delta_opd_fin[0]
                
            # Get final RMS in nm
            rms_tot = np.asarray(self.calc_rms(delta_opd_fin)) * 1e9
        else:
            rms_tot = np.sqrt(amp_thermal**2 + amp_frill**2 + amp_iec**2)
        
        wfe_amps = {
            'thermal': amp_thermal,
            'frill'  : amp_frill,
            'iec'    : amp_iec,
            'total'  : rms_tot
        }
        
        if return_wfe_amps and return_dopd_fin:
            return delta_opd_fin, wfe_amps
        elif return_wfe_amps:
            return wfe_amps
        elif return_dopd_fin:
            return delta_opd_fin
        else:
            _log.warning('Must specify `return_wfe_amps` and/or `return_dopd_fin`')

            
    
    def evolve_dopd(self, delta_time, slew_angles, case='BOL', 
                   return_wfe_amps=True, return_dopd_fin=True, 
                   do_thermal=True, do_frill=True, do_iec=True, **kwargs):
        
        """ Evolve the delta OPD with multiple slews
        
        Input an array of `delta_time` and `slew_angles` to return the 
        evolution of a delta_OPD image. Option to return the various
        WFE components, including OTE backplane (thermal), frill tensioning,
        and IEC heater switching.
        
        Parameters
        ----------
        delta_time : astropy.units quantity object
            An array of times assuming astropy units.
        slew_angles : ndarray
            The sun pitch angles, in degrees between -5 and +45.
        case : string
            Either "BOL" for current best estimate at beginning of life, or
            "EOL" for more conservative prediction at end of life.
        do_thermal : bool
            Include thermal slew component? Mostly for debugging purposes.
        do_frill : bool
            Include frill component? Mostly for debugging purposes.
        do_iec : bool
            Include IEC component? Good to exclude if calling this function
            repeatedly for evolution of multiple slews, then add IEC later.
        return_wfe_amps : bool
            Return a dictionary that provides the RMS WFE (nm) of each
            component at each time step.
        return_dopd_fin : bool
            Option to exclude calculating final delta OPD in case we only
            want the final RMS WFE dictionary.
        
        Keyword Args
        ------------
        amplitude : float
            Full amplitude of IEC arcsine distribution. Values will range
            from -0.5*amplitude to +0.5*amplitude.
        period : float
            Period in minutes of IEC oscillations. Usually 3-5 minutes.
        random_seed : int
            Random seed to pass to IEC generation.
        """

        if (not return_wfe_amps) and (not return_dopd_fin):
            _log.warning('Must specify `return_wfe_amps` and/or `return_dopd_fin`')
            return

        log_prev = conf.logging_level
        if 'WARN' not in log_prev:
            setup_logging('WARN', verbose=False)

        # Indices where slews occur
        islew = np.where(slew_angles[1:] - slew_angles[:-1] != 0)[0] + 1
        islew = np.concatenate(([0], islew))

        # Build delta OPDs for each slew angle
        kwargs['case'] = case
        kwargs['return_wfe_amps'] = return_wfe_amps
        kwargs['return_dopd_fin'] = return_dopd_fin
        kwargs['do_thermal'] = do_thermal
        kwargs['do_frill'] = do_frill
        kwargs['do_iec'] = False
        iter_func = tqdm(islew, desc='Slews') if return_dopd_fin else islew
        for i in iter_func:
            ang1 = slew_angles[0] if i==0 else ang2
            ang2 = slew_angles[i]

            tvals = delta_time[i:]
            tvals = tvals - tvals[0]
            res = self.gen_delta_opds(tvals, start_angle=ang1, end_angle=ang2, **kwargs)

            if return_wfe_amps and return_dopd_fin:
                dopds, wfe_dict = res
            elif return_dopd_fin:
                dopds = res
            elif return_wfe_amps:
                wfe_dict = res
                
            # Accumulate delta OPD images
            if return_dopd_fin:
                if i==0:
                    dopds_fin = dopds + 0.0
                else:
                    dopds_fin[i:] += dopds
                del dopds

            # Add in drift amplitudes for thermal and frill components
            if return_wfe_amps:
                if i==0:
                    wfe_dict_fin = wfe_dict
                else:
                    for k in wfe_dict.keys():
                        wfe_dict_fin[k][i:] += wfe_dict[k]
                        
        # Get IEC values
        if do_iec:
            kwargs['do_thermal'] = False
            kwargs['do_frill'] = False
            kwargs['do_iec'] = True
            res = self.gen_delta_opds(delta_time-delta_time[0], **kwargs)
            
            if return_wfe_amps and return_dopd_fin:
                dopds, wfe_dict = res
                wfe_dict_fin['iec'] = wfe_dict['iec']
            elif return_dopd_fin:
                dopds = res
            elif return_wfe_amps:
                wfe_dict = res
                wfe_dict_fin['iec'] = wfe_dict['iec']

            # Add IEC OPDs
            if return_dopd_fin:
                dopds_fin += dopds
                del dopds
            
        if 'WARN' not in log_prev:
            setup_logging(log_prev, verbose=False)

        # Calculate RMS values on final delta OPDs
        if return_wfe_amps:
            if return_dopd_fin:
                rms_tot = self.calc_rms(dopds_fin)*1e9
            else:
                rms_tot = np.sqrt(wfe_dict_fin['thermal']**2 + wfe_dict_fin['frill']**2 + wfe_dict_fin['iec']**2)
            wfe_dict_fin['total'] = rms_tot

        if return_wfe_amps and return_dopd_fin:
            return dopds_fin, wfe_dict_fin
        elif return_dopd_fin:
            return dopds_fin
        elif return_wfe_amps:
            return wfe_dict_fin


    def interp_dopds(self, delta_time, dopds, dt_new, wfe_dict=None, interp_kind='linear', **kwargs):
        """ Interpolate an array of delta OPDs

        Perform a linear interpolation on a series of delta OPDS.

        Parameters
        ----------
        delta_time : astropy.units quantity object
            An array of times assuming astropy units corresponding to each `dopd`.
        dopds : ndarray
            Array of delta OPD images associated with `delta_time`.
        dt_new : astropy.units quantity object
            New array to interpolate onto.

        Keyword Args
        ------------
        wfe_dict : dict or None
            If specified, then must provide a dictionary where the values
            for each keywords are the WFE drift components associated with
            each `delta_time`. Will then return a dictionary 
        interp_kind : str or int
            Specifies the kind of interpolation as a string
            ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic',
            'previous', 'next', where 'zero', 'slinear', 'quadratic' and 'cubic'
            refer to a spline interpolation of zeroth, first, second or third
            order; 'previous' and 'next' simply return the previous or next value
            of the point) or as an integer specifying the order of the spline
            interpolator to use.
            Default is 'linear'.

        """
        dt_new_vals = dt_new.to('hour')

        # Create interpolation function
        dt_vals = delta_time.to('hour')
        func = interp1d(dt_vals, dopds, axis=0, kind=interp_kind, bounds_error=True)

        opds_new = func(dt_new_vals)

        if wfe_dict is not None:
            wfe_dict_new = {}
            for k in wfe_dict.keys():
                vals = wfe_dict[k]
                func = interp1d(dt_vals, vals, kind=interp_kind, bounds_error=True)
                wfe_dict_new[k] = func(dt_new_vals)

            return opds_new, wfe_dict_new

        else:
            return opds_new

        
    def slew_pos_averages(self, delta_time, slew_angles, opds=None, wfe_dict=None, 
                          mn_func=np.mean, interpolate=False, **kwargs):
        """ Get averages at each slew position

        Given a series of times and slew angles, calculate the average OPD and
        WFE RMS error within each slew angle position. Returns a tuple with new 
        arrays of (dt_new, opds_new, wfe_dict_new). 
        
        If input both `opds` and `wfe_dict` are not specified, then we call
        the `evolve_dopd` function and return .
        
        Parameters
        ----------
        delta_time : astropy.units quantity object
            An array of times assuming astropy units.
        slew_angles : ndarray
            The sun pitch angles at each `delta_time`, in degrees between -5 and +45.
        opds : ndarray or None
            Cube of OPD images (or delta OPDs) associated with each `delta_time`.
            If set to None, then a new set of OPDs are not calculated.
        wfe_dict : dict or None
            If specified, then must provide a dictionary where the values
            for each keywords are the WFE drift components associated with
            each `delta_time`. New set of WFE dictionary is not calculated if set 
            to None.
        mn_func : function
            Function to use for taking averages. Default: np.mean()
        interpolate : bool
            Instead of taking average, use the interpolation function `self.interp_dopds()`.
            
        Keyword Args
        ------------
        case : string
            Either "BOL" for current best estimate at beginning of life, or
            "EOL" for more conservative prediction at end of life.
        do_thermal : bool
            Include thermal slew component? Mostly for debugging purposes.
        do_frill : bool
            Include frill component? Mostly for debugging purposes.
        do_iec : bool
            Include IEC component? Good to exclude if calling this function
            repeatedly for evolution of multiple slews, then add IEC later.
        amplitude : float
            Full amplitude of IEC arcsine distribution. Values will range
            from -0.5*amplitude to +0.5*amplitude.
        period : float
            Period in minutes of IEC oscillations. Usually 3-5 minutes.
        kind : str or int
            Specifies the kind of interpolation (if specified) as a string.
            Default: 'linear'.

        """
        
        if (opds is None) and (wfe_dict is None):
            kwargs['return_wfe_amps'] = True
            kwargs['return_dopd_fin'] = True
            opds, wfe_dict = self.evolve_dopd(delta_time, slew_angles, **kwargs)

        # Indices where slews occur
        islew = np.where(slew_angles[1:] - slew_angles[:-1] != 0)[0] + 1

        # Start and stop indices for each slew position
        i1_arr = np.concatenate(([0], islew))
        i2_arr = np.concatenate((islew, [len(slew_angles)]))

        # Get average time at each position
        dt_new = np.array([mn_func(delta_time[i1:i2].value) for i1, i2 in zip(i1_arr, i2_arr)]) 
        dt_new = dt_new * delta_time.unit

        if interpolate:
            res = self.interp_dopds(delta_time, opds, dt_new, wfe_dict=wfe_dict, **kwargs)
            if wfe_dict is None:
                opds_new = res
                wfe_dict_new = None
            else:
                opds_new, wfe_dict_new = res
            return dt_new, opds_new, wfe_dict_new
        
        # Averages of OPD at each position
        if opds is not None:
            opds_new = np.array([mn_func(opds[i1:i2], axis=0) for i1, i2 in zip(i1_arr, i2_arr)])
        else:
            opds_new = None

        # Get average of each WFE drift component
        if wfe_dict is not None:
            wfe_dict_new = {}
            for k in wfe_dict.keys():
                wfe_dict_new[k] = np.array([mn_func(wfe_dict[k][i1:i2]) for i1, i2 in zip(i1_arr, i2_arr)])
            if opds_new is not None:
                wfe_dict_new['total'] = self.calc_rms(opds_new)*1e9
        else:
            wfe_dict = None

        return dt_new, opds_new, wfe_dict_new
    
    
    def opds_as_hdul(self, delta_time, slew_angles, delta_opds=None, wfe_dict=None,
                     case=None, add_main_opd=True, slew_averages=False, 
                     return_ind=None, **kwargs):
        """Convert series of delta OPDS to HDUList"""

        if delta_opds is None:
            case = 'BOL' if case is None else case
            kwargs['case'] = case
            kwargs['return_wfe_amps'] = True
            kwargs['return_dopd_fin'] = True
            delta_opds, wfe_dict = self.evolve_dopd(delta_time, slew_angles, **kwargs)
            
        if slew_averages:
            res = self.slew_pos_averages(delta_time, slew_angles, opds=delta_opds, 
                                         wfe_dict=wfe_dict, **kwargs)
            delta_time, delta_opds, wfe_dict = res
            # Indices where slews occur
            islew = np.where(slew_angles[1:] - slew_angles[:-1] != 0)[0] + 1
            islew = np.concatenate(([0], islew))
            slew_angles = slew_angles[islew]

        nz, ny, nx = delta_opds.shape

        # Indices where slews occur
        islew = np.where(slew_angles[1:] - slew_angles[:-1] != 0)[0] + 1
        islew = np.concatenate(([0], islew))

        hdul = fits.HDUList()
        for i in range(nz):
            if i<islew[1]:
                ang1 = ang2 = slew_angles[i]
            else:
                if i in islew:
                    ang1 = slew_angles[i-1]
                ang2 = slew_angles[i]

            # Skip if only returning a single OPD
            if (return_ind is not None) and (i != return_ind):
                continue

            # Update header
            dt = delta_time[i].to(u.day).to_string()

            opd_im = self._opd_original + delta_opds[i] if add_main_opd else delta_opds[i]

            hdu = fits.ImageHDU(data=opd_im, header=self.opd_header, name=f'OPD{i}')
            hdr = hdu.header
            hdr['BUNIT']    = 'meter'
            hdr['DELTA_T']  = (dt, "Delta time after initial slew [d]")
            hdr['STARTANG'] = (ang1, "Starting sun pitch angle [deg]")
            hdr['ENDANG']   = (ang2, "Ending sun pitch angle [deg]")
            hdr['THRMCASE'] = (case, "Thermal model case, beginning or end of life")
            # if add_main_opd:
            #     hdr['OPDSLICE'] = (self.opd_slice, 'OPD slice index')

            hdr['WFE_RMS'] = (self.calc_rms(hdu.data)*1e9, "RMS WFE [nm]")
            # Include the WFE RMS inputs from each component
            if wfe_dict is not None:
                for k in wfe_dict.keys():
                    hdr[k] = (wfe_dict[k][i], f"{k} RMS delta WFE [nm]")

            hdul.append(hdu)

        return hdul
    
    
def plot_im(im, fig, ax, vlim=None, add_cbar=True, return_ax=False, 
    extent=None, cmap='RdBu_r'):
    """
    Plot single image on some axes
    """
    
    if vlim is None:
        vlim = np.max(np.abs(im))
        
    img = ax.imshow(im, cmap=cmap, vmin=-1*vlim, vmax=+1*vlim, extent=extent)
    
    # Add colorbar
    if add_cbar:
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Amplitude [nm]')

    if return_ax and add_cbar:
        return ax, cbar
    elif return_ax:
        return ax
    

def plot_opd(hdul, index=1, opd0=None, vlim1=None, vlim2=None):
    """ 
    Plot OPDs images (full or delta)
    """
    
    def calc_rms_nm(im):
        ind = (im != 0) & (np.isfinite(im))
        rms = np.sqrt((im[ind] ** 2).mean()) * 1e9
        return rms

    m_to_nm = 1e9
    
    # Define OPD to compare delta OPD image
    opd0 = hdul[0].data if opd0 is None else opd0
    
    # Header and data for current image
    header = hdul[index].header
    opd = hdul[index].data    
    opd_diff = (opd - opd0)
    
    rms_opd = calc_rms_nm(opd)
    rms_diff = calc_rms_nm(opd_diff)

    # Time since slew
    delta_time = header['DELTA_T']
    
    try:
        pupilscale = header['PUPLSCAL']
        s = opd.shape
        extent = [a * pupilscale for a in [-s[0] / 2, s[0] / 2, -s[1] / 2, s[1] / 2]]
    except KeyError:
        extent = None
    
    # Create figure
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    
    ax = axes[0]
    vlim = 3*rms_opd if vlim1 is None else vlim1
    plot_im(opd * m_to_nm, fig, ax, vlim=vlim, extent=extent)
    data_val, data_units = str.split(delta_time)
    
    data_val = np.float(data_val)
    if 'h' in data_units:
        dt = data_val * u.hr
    elif 'm' in data_units:
        dt = data_val * u.min
    elif 'd' in data_units:
        dt = data_val * u.day
    # Convert to hours
    dt = dt.to('hr')
    
    ax.set_title("Delta Time = {:.1f} (RMS = {:.2f} nm)".format(dt, rms_opd))
    
    ax = axes[1]
    vlim = 3*rms_diff if vlim2 is None else vlim2
    plot_im(opd_diff * m_to_nm, fig, ax, vlim=vlim, extent=extent)
    ax.set_title("Delta OPD = {:.2f} nm RMS".format(rms_diff))
    
    fig.tight_layout()

    plt.draw()


def slew_time(dist_asec):
    """
    Given a slew distance (arcsec), calculate telescope slew time. Output is sec.
    Data comes from JDOX website:

        https://jwst-docs.stsci.edu/jppom/visit-overheads-timing-model/slew-times.
    """
    # Slew value in arcsec
    slew_arr = np.array([
        0, 0.06, 0.0600001, 15, 20, 20.0000001, 30, 50,
        100, 150, 300, 1000, 3600, 4000, 10000, 10800,
        10800, 14400, 18000, 21600, 25200, 28800, 32400, 36000, 39600,
        43200, 46800, 50400, 54000, 57600, 61200, 64800, 68400, 72000,
        108000, 144000, 180000, 216000, 252000, 288000, 324000, 360000, 
        396000, 432000, 468000, 504000, 540000, 576000, 612000, 648000
    ])
    # Slew times
    tsec_arr = np.array([
        0, 0, 20.48, 20.48, 23.296, 101.632, 116.224, 137.728,
        173.568, 198.656, 250.112, 373.504, 572.416, 592.896, 804.864, 825.6, 521.216, 
        578.048, 628.608, 674.56, 716.928, 756.608, 793.856, 829.184, 862.848, 894.976,
        925.824, 955.648, 984.32, 1012.224, 1039.104, 1065.344, 1090.816, 1115.648,
        1336.448, 1537.408,1744, 1939.328, 2112.192, 2278.272, 2440.32, 2599.936, 
        2757.632, 2914.24, 3069.888, 3224.832, 3379.328, 3533.376, 3687.104, 3840.512
    ])

    return np.interp(dist_asec, slew_arr, tsec_arr)

