# Import libraries
from operator import add
import numpy as np

import time
import os, six
from pathlib import Path

import multiprocessing as mp
import traceback

from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from copy import deepcopy

# Bandpasses, PSFs, and OPDs
from .bandpasses import miri_filter, nircam_filter
from .psfs import nproc_use, gen_image_from_coeff
from .psfs import make_coeff_resid_grid, field_coeff_func
from .opds import OPDFile_to_HDUList
from .spectra import stellar_spectrum

# Coordinates and image manipulation
from .coords import NIRCam_V2V3_limits, xy_rot, xy_to_rtheta, rtheta_to_xy
from .image_manip import frebin, pad_or_cut_to_size, rotate_offset
from .image_manip import fourier_imshift, fshift

# Polynomial fitting routines
from .maths import jl_poly, jl_poly_fit
import scipy
from scipy.interpolate import interp1d


# Logging info
from . import conf
from .logging_utils import setup_logging
import logging
_log = logging.getLogger('webbpsf_ext')

from .version import __version__

# WebbPSF and Poppy stuff
from .utils import check_fitsgz, webbpsf, poppy, S
from webbpsf import MIRI as webbpsf_MIRI
from webbpsf import NIRCam as webbpsf_NIRCam
from webbpsf.opds import OTE_Linear_Model_WSS

# Program bar
from tqdm.auto import trange, tqdm

# NIRCam Subclass
class NIRCam_ext(webbpsf_NIRCam):

    """ NIRCam instrument PSF coefficients
    
    Subclass of WebbPSF's NIRCam class for generating polynomial coefficients
    to cache and quickly generate PSFs for arbitrary spectral types as well
    as WFE variations due to field-dependent OPDs and telescope thermal drifts.

    Parameters
    ==========
    filter : str
        Name of input filter.
    pupil_mask : str, None
        Pupil elements such as grisms or lyot stops (default: None).
    image_mask : str, None
        Specify which coronagraphic occulter (default: None).
    fov_pix : int
        Size of the PSF FoV in pixels (real SW or LW pixels).
        The defaults depend on the type of observation.
        Odd number place the PSF on the center of the pixel,
        whereas an even number centers it on the "crosshairs."
    oversample : int
        Factor to oversample during WebbPSF calculations.
        Default 2 for coronagraphy and 4 otherwise.
    """

    def __init__(self, filter=None, pupil_mask=None, image_mask=None, 
                 fov_pix=None, oversample=None, **kwargs):
        
        webbpsf_NIRCam.__init__(self)

        # Initialize script
        _init_inst(self, filter=filter, pupil_mask=pupil_mask, image_mask=image_mask,
                   fov_pix=fov_pix, oversample=oversample, **kwargs)

        # No jitter for coronagraphy by default
        # Otherwise, assume 5 mas / axis
        self.options['jitter'] = None if self.is_coron else 'gaussian'
        self.options['jitter_sigma'] = 0.005

        # By default, WebbPSF has wavelength limits depending on the channel
        # which can interfere with coefficient calculations, so set these to 
        # extreme low/high values across the board.
        self.SHORT_WAVELENGTH_MIN = self.LONG_WAVELENGTH_MIN = 1e-7
        self.SHORT_WAVELENGTH_MAX = self.LONG_WAVELENGTH_MAX = 10e-6

        # Detector name to SCA ID
        self._det2sca = {
            'A1':481, 'A2':482, 'A3':483, 'A4':484, 'A5':485,
            'B1':486, 'B2':487, 'B3':488, 'B4':489, 'B5':490,
        }

        # Option to use 1st or 2nd order for grism bandpasses
        self._grism_order = 1

        # Specify ice and nvr scalings
        self._ice_scale = kwargs['ice_scale'] if 'ice_scale' in kwargs.keys() else None
        self._nvr_scale = kwargs['nvr_scale'] if 'nvr_scale' in kwargs.keys() else None
        self._ote_scale = kwargs['ote_scale'] if 'ote_scale' in kwargs.keys() else None
        self._nc_scale  = kwargs['nc_scale']  if 'nc_scale'  in kwargs.keys() else None

        # Option to calculate ND acquisition for coronagraphic obs
        self._ND_acq = False
        # Option to specify that coronagraphic substrate materials is present.
        self._coron_substrate = None

    @property
    def save_dir(self):
        """Coefficient save directory"""
        return _gen_save_dir(self)
    @save_dir.setter
    def save_dir(self, value):
        self._save_dir = value
    def _erase_save_dir(self):
        """Erase all instrument coefficients"""
        _clear_coeffs_dir(self)

    @property
    def save_name(self):
        """Coefficient file name"""
        if self._save_name is None:
            return self.gen_save_name()
        else:
            return self._save_name
    @save_name.setter
    def save_name(self, value):
        self._save_name = value

    @property
    def is_lyot(self):
        """
        Is a Lyot mask in the pupil wheel?
        """
        pupil = self.pupil_mask
        return (pupil is not None) and ('LYOT' in pupil)
    @property
    def is_coron(self):
        """
        Observation with coronagraphic mask (incl Lyot stop)?
        """
        mask = self.image_mask
        return self.is_lyot and ((mask is not None) and ('MASK' in mask))
    @property
    def is_grism(self):
        pupil = self.pupil_mask
        return (pupil is not None) and ('GRISM' in pupil)
    @property
    def is_dark(self):
        pupil = self.pupil_mask
        return (pupil is not None) and ('FLAT' in pupil)

    @property
    def ND_acq(self):
        """Use Coronagraphic ND acquisition square?"""
        return self._ND_acq
    @ND_acq.setter
    def ND_acq(self, value):
        _check_list(value, [True, False], 'ND_acq')
        self._ND_acq = value

    @property
    def coron_substrate(self):
        """Include coronagraphic substrate material?"""
        # True by default if Lyot stop is in place.
        # User should override this if intention is to get 
        # PSF outside of substrate region.
        if ((self._coron_substrate is None) and self.is_lyot):
            val = True
        else: 
            val = self._coron_substrate
        return val
    @coron_substrate.setter
    def coron_substrate(self, value):
        _check_list(value, [True, False], 'coron_substrate')
        self._coron_substrate = value

    @property
    def fov_pix(self):
        return self._fov_pix
    @fov_pix.setter
    def fov_pix(self, value):
        self._fov_pix = value
        
    @property
    def oversample(self):
        if self._oversample is None:
            # Detector oversampling of 2 if coronagraphy
            # Calculation will still occur w/ FFT oversampling of 4 
            oversample = 2 if self.is_lyot else 4
        else:
            oversample = self._oversample
        return oversample
    @oversample.setter
    def oversample(self, value):
        self._oversample = value
    
    @property
    def wave_fit(self):
        """Wavelength range to fit"""
        if self.quick:
            w1 = self.bandpass.wave.min() / 1e4
            w2 = self.bandpass.wave.max() / 1e4
        else:
            w1, w2 = (2.4,5.2) if self.channel=='long' else (0.5,2.5)
        return w1, w2
    @property
    def npsf(self):
        """Number of wavelengths/PSFs to fit"""
        w1, w2 = self.wave_fit
        npsf = self._npsf
        # Default to number of PSF simulations per um
        if npsf is None:
            dn = 10 if self.channel=='long' else 20
            npsf = int(np.ceil(dn * (w2-w1)))
            
        # Want at least 5 monochromatic PSFs
        npsf = 5 if npsf<5 else int(npsf)

        # Number of points must be greater than degree of fit
        npsf = self.ndeg+1 if npsf<=self.ndeg else int(npsf)

        return npsf
    @npsf.setter
    def npsf(self, value):
        """Set number of wavelengths/PSFs to fit"""
        self._npsf = value
        
    @property
    def ndeg(self):
        ndeg = self._ndeg
        if ndeg is None:
            if self.quick:
                if  self.filter[-2:]=='W2':
                    ndeg = 9
                elif self.filter[-1]=='W':
                    ndeg = 8
                elif self.filter[-1]=='M':
                    ndeg = 6
                elif self.filter[-1]=='N':
                    ndeg = 4
                else:
                    raise ValueError(f'{self.filter} not recognized as narrow, medium, wide, or double wide.')
            else:
                ndeg = 9
        return ndeg
    @ndeg.setter
    def ndeg(self, value):
        self._ndeg = value
    @property
    def quick(self):
        """Perform quicker coeff calculation over limited bandwidth?"""

        # If quick is not explicitly set by user, then default to True:
        if self._quick is not None:
            quick = self._quick
        else:
            quick = True
            
        return quick
    @quick.setter
    def quick(self, value):
        """Perform quicker coeff calculation over limited bandwidth?"""
        _check_list(value, [True, False], 'quick')
        self._quick = value

    @webbpsf_NIRCam.pupil_mask.setter
    def pupil_mask(self, name):

        if name != self._pupil_mask:
            # only apply updates if the value is in fact new
            if name=='GRISM0':
                name = 'GRISMR'
            elif name=='GRISM90':
                name = 'GRISMC'
            super(NIRCam_ext, self.__class__).pupil_mask.__set__(self, name)

    @property
    def siaf_ap(self):
        """SIAF Aperture object"""
        if self._siaf_ap is None:
            return self.siaf[self.aperturename]
        else:
            return self._siaf_ap
    @siaf_ap.setter
    def siaf_ap(self, value):
        self._siaf_ap = value

    @property
    def scaid(self):
        """SCA ID (481, 482, ... 489, 490)"""
        detid = self.detector[-2:]
        return self._det2sca.get(detid, 'unknown')
    @scaid.setter
    def scaid(self, value):
        scaid_values = np.array(list(self._det2sca.values()))
        det_values = np.array(list(self._det2sca.keys()))
        if value in scaid_values:
            ind = np.where(scaid_values==value)[0][0]
            self.detector = 'NRC'+det_values[ind]
        else:
            _check_list(value, scaid_values, var_name='scaid')


    @webbpsf_NIRCam.detector_position.setter
    def detector_position(self, position):
        # Remove limits for detector position
        # Values outside of [0,2047] will get transformed to the correct V2/V3 location
        try:
            x, y = map(float, position)
        except ValueError:
            raise ValueError("Detector pixel coordinates must be a pair of numbers, not {}".format(position))
        self._detector_position = (x,y)

    def _get_fits_header(self, result, options):
        """ populate FITS Header keywords """
        super(NIRCam_ext, self)._get_fits_header(result, options)

        # Keep detector X and Y positions as floats
        dpos = np.asarray(self.detector_position, dtype=float)
        result[0].header['DET_X'] = (dpos[0], "Detector X pixel position of array center")
        result[0].header['DET_Y'] = (dpos[1], "Detector Y pixel position of array center")

    @property
    def fastaxis(self):
        """Fast readout direction in sci coords"""
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#orientation-of-detector-image
        # 481, 3, 5, 7, 9 have fastaxis equal -1
        # Others have fastaxis equal +1
        fastaxis = -1 if np.mod(self.scaid,2)==1 else +1
        return fastaxis
    @property
    def slowaxis(self):
        """Slow readout direction in sci coords"""
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#orientation-of-detector-image
        # 481, 3, 5, 7, 9 have slowaxis equal +2
        # Others have slowaxis equal -2
        slowaxis = +2 if np.mod(self.scaid,2)==1 else -2
        return slowaxis

    @property
    def bandpass(self):
        """ Return bandpass throughput """
        try:
            kwargs = {
                'ice_scale': self._ice_scale,
                'nvr_scale': self._nvr_scale,
                'ote_scale': self._ote_scale,
                'nc_scale' : self._nc_scale,
            }            
            bp = nircam_filter(self.filter, pupil=self.pupil_mask, mask=self.image_mask,
                                module=self.module, grism_order=self._grism_order,
                                ND_acq=self.ND_acq, coron_substrate=self.coron_substrate, 
                                **kwargs)
        except AttributeError:
            bp = nircam_filter(self.filter, pupil=self.pupil_mask, mask=self.image_mask,
                               module=self.module)
        
        return bp

    def get_bar_offset(self, narrow=False):
        """
        Obtain the value of the bar offset that would be passed through to
        PSF calculations for bar/wedge coronagraphic masks.
        """

        from webbpsf.optics import NIRCam_BandLimitedCoron

        if (self.is_coron) and (self.image_mask[-1:]=='B'):
            # Determine bar offset for Wedge masks either based on filter 
            # or explicit specification
            bar_offset = self.options.get('bar_offset', None)
            if bar_offset is None:
                auto_offset = 'narrow' if narrow else self.filter
            else:
                try:
                    _ = float(bar_offset)
                    auto_offset = None
                except ValueError:
                    # If the "bar_offset" isn't a float, pass it to auto_offset instead
                    auto_offset = bar_offset
                    bar_offset = None

            mask = NIRCam_BandLimitedCoron(name=self.image_mask, module=self.module, kind='nircamwedge',
                                           bar_offset=bar_offset, auto_offset=auto_offset)
            return mask.bar_offset
        else:
            return None

    def gen_mask_image(self, npix=None, pixelscale=None, bar_offset=None, nd_squares=True):
        """
        Return an image representation of the focal plane mask attenuation.
        Output is in 'sci' coords orientation. If no image mask
        is present, then returns an array of all 1s. Mask is
        centered in image, while actual subarray readout has a
        slight offset.
        
        Parameters
        ==========
        npix : int
            Number of pixels in output image. If not set, then
            is automatically determined based on mask FoV and
            `pixelscale`
        pixelscale : float
            Size of output pixels in units of arcsec. If not specified,
            then selects oversample pixel scale.
        """
        
        from webbpsf.optics import NIRCam_BandLimitedCoron

        shifts = {'shift_x': self.options.get('coron_shift_x', None),
                  'shift_y': self.options.get('coron_shift_y', None)}

        if pixelscale is None:
            pixelscale = self.pixelscale / self.oversample
        if npix is None:
            osamp = self.pixelscale / pixelscale
            npix = 320 if 'long' in self.channel else 640
            npix = int(npix * osamp + 0.5)

        if self.is_coron:
            if self.image_mask[-1:]=='B':
                bar_offset = self.get_bar_offset() if bar_offset is None else bar_offset
            else:
                bar_offset = None
            mask = NIRCam_BandLimitedCoron(name=self.image_mask, module=self.module, nd_squares=nd_squares,
                                           bar_offset=bar_offset, auto_offset=None, **shifts)

            # Create wavefront to pass through mask and obtain transmission image
            wavelength = self.bandpass.avgwave() / 1e10
            wave = poppy.Wavefront(wavelength=wavelength, npix=npix, pixelscale=pixelscale)
            im = mask.get_transmission(wave)**2
        else:
            im = np.ones([npix,npix])

        return im

    def get_opd_info(self, opd=None, pupil=None, HDUL_to_OTELM=True):
        """
        Parse out OPD information for a given OPD, which can be a file name, tuple (file,slice), 
        HDUList, or OTE Linear Model. Returns dictionary of some relevant information for 
        logging purposes. The dictionary returns the OPD as an OTE LM by default.
        
        This outputs an OTE Linear Model. In order to update instrument class:
            >>> opd_dict = inst.get_opd_info()
            >>> opd_new = opd_dict['pupilopd']
            >>> inst.pupilopd = opd_new
            >>> inst.pupil = opd_new
        """
        return _get_opd_info(self, opd=opd, pupil=pupil, HDUL_to_OTELM=HDUL_to_OTELM)

    def drift_opd(self, wfe_drift, opd=None):
        """
        A quick method to drift the pupil OPD. This function applies some WFE drift to input 
        OPD file by breaking up the wfe_drift attribute into thermal, frill, and IEC components. 
        If we want more realistic time evolution, then we should use the procedure in 
        dev_utils/WebbPSF_OTE_LM.ipynb to create a time series of OPD maps, which can then be 
        passed directly to create unique PSFs.
        
        This outputs an OTE Linear Model. In order to update instrument class:
            >>> opd_dict = inst.drift_opd()
            >>> inst.pupilopd = opd_dict['opd']
            >>> inst.pupil = opd_dict['opd']
        """
        return _drift_opd(self, wfe_drift, opd=opd)

    def gen_save_name(self, wfe_drift=0):
        """
        Generate save name for polynomial coefficient output file.
        """
        return _gen_save_name(self, wfe_drift=wfe_drift)

    def gen_psf_coeff(self, bar_offset=0, **kwargs):
        """Generate PSF coefficients

        Creates a set of coefficients that will generate simulated PSFs for any
        arbitrary wavelength. This function first simulates a number of evenly-
        spaced PSFs throughout the specified bandpass (or the full channel). 
        An nth-degree polynomial is then fit to each oversampled pixel using 
        a linear-least squares fitting routine. The final set of coefficients 
        for each pixel is returned as an image cube. The returned set of 
        coefficient are then used to produce PSF via `calc_psf_from_coeff`.

        Useful for quickly generated imaging and dispersed PSFs for multiple
        spectral types. 

        Parameters
        ----------
        bar_offset : float
            For wedge masks, option to set the PSF position across the bar.
            In this framework, we generally set the default to 0, then use
            the `gen_wfemask_coeff` function to determine how the PSF changes 
            along the wedge axis as well as perpendicular to the wedge. This
            allows for more arbitrary PSFs within the mask, including small
            grid dithers as well as variable PSFs for extended objects.
            Default: 0. 

        Keyword Args
        ------------
        wfe_drift : float
            Wavefront error drift amplitude in nm.
        force : bool
            Forces a recalculation of PSF even if saved PSF exists. (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)
        nproc : bool or None
            Manual setting of number of processor cores to break up PSF calculation.
            If set to None, this is determined based on the requested PSF size,
            number of available memory, and hardware processor cores. The automatic
            calculation endeavors to leave a number of resources available to the
            user so as to not crash the user's machine. 
        return_results : bool
            By default, results are saved as object the attributes `psf_coeff` and
            `psf_coeff_header`. If return_results=True, results are instead returned
            as function outputs and will not be saved to the attributes. This is mostly
            used for successive coeff simulations to determine varying WFE drift or 
            focal plane dependencies.
        return_extras : bool
            Additionally returns a dictionary of monochromatic PSFs images and their 
            corresponding wavelengths for debugging purposes. Can be used with or without
            `return_results`. If `return_results=False`, then only this dictionary is
            returned, otherwise if `return_results=False` then returns everything as a
            3-element tuple (psf_coeff, psf_coeff_header, extras_dict).
        """

        # Set to input bar offset value. No effect if not a wedge mask.
        bar_offset_orig = self.options.get('bar_offset', None)
        self.options['bar_offset'] = bar_offset
        res = _gen_psf_coeff(self, **kwargs)
        self.options['bar_offset'] = bar_offset_orig

        return res

    def gen_wfedrift_coeff(self, force=False, save=True, **kwargs):
        """ Fit WFE drift coefficients

        This function finds a relationship between PSF coefficients in the 
        presence of WFE drift. For a series of WFE drift values, we generate 
        corresponding PSF coefficients and fit a  polynomial relationship to 
        the residual values. This allows us to quickly modify a nominal set of 
        PSF image coefficients to generate a new PSF where the WFE has drifted 
        by some amplitude.
        
        It's Legendre's all the way down...

        Parameters
        ----------
        force : bool
            Forces a recalculation of coefficients even if saved file exists. 
            (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)

        Keyword Args
        ------------
        wfe_list : array-like
            A list of wavefront error drift values (nm) to calculate and fit.
            Default is [0,1,2,5,10,20,40], which covers the most-likely
            scenarios (1-5nm) while also covering a range of extreme drift
            values (10-40nm).
        return_results : bool
            By default, results are saved in `self._psf_coeff_mod` dictionary. 
            If return_results=True, results are instead returned as function outputs 
            and will not be saved to the dictionary attributes. 
        return_raw : bool
            Normally, we return the relation between PSF coefficients as a function
            of position. Instead this returns (as function outputs) the raw values
            prior to fitting. Final results will not be saved to the dictionary attributes.
        """

        # Set to input bar offset value. No effect if not a wedge mask.
        bar_offset_orig = self.options.get('bar_offset', None)
        try:
            self.options['bar_offset'] = self.psf_coeff_header.get('BAROFF', None)
        except AttributeError:
            # Throws error if psf_coeff_header doesn't exist
            _log.error("psf_coeff_header does not appear to exist. Run gen_psf_coeff().")
            res = 0
        else:
            res = _gen_wfedrift_coeff(self, force=force, save=save, **kwargs)
        finally:
            self.options['bar_offset'] = bar_offset_orig

        return res

    def gen_wfemask_coeff(self, large_grid=False, force=False, save=True, **kwargs):
        """ Fit WFE changes in mask position

        For coronagraphic masks, slight changes in the PSF location
        relative to the image plane mask can substantially alter the 
        PSF speckle pattern. This function generates a number of PSF
        coefficients at a variety of positions, then fits polynomials
        to the residuals to track how the PSF changes across the mask's
        field of view. Special care is taken near the 10-20mas region
        in order to provide accurate sampling of the SGD offsets. 

        Parameters
        ----------
        large_grid : bool
            Use a large number (high-density) of grid points to create coefficients.
            If True, then a higher fidelity PSF variations across the FoV, but could
            take hours to generate on the first pass. Setting to False allows for
            quicker coefficient creation with a smaller memory footprint, useful for
            testing and debugging.        
        force : bool
            Forces a recalculation of coefficients even if saved file exists. 
            (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)

        Keyword Args
        ------------
        return_results : bool
            By default, results are saved in `self._psf_coeff_mod` dictionary. 
            If return_results=True, results are instead returned as function outputs 
            and will not be saved to the dictionary attributes. 
        return_raw : bool
            Normally, we return the relation between PSF coefficients as a function
            of position. Instead this returns (as function outputs) the raw values
            prior to fitting. Final results will not be saved to the dictionary attributes.

        """
        return _gen_wfemask_coeff(self, large_grid=large_grid, force=force, save=save, **kwargs)

    def gen_wfefield_coeff(self, force=False, save=True, **kwargs):
        """ Fit WFE field-dependent coefficients

        Find a relationship between field position and PSF coefficients for
        non-coronagraphic observations and when `include_si_wfe` is enabled.

        Parameters
        ----------
        force : bool
            Forces a recalculation of coefficients even if saved file exists. 
            (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)
            
        Keyword Args
        ------------
        return_results : bool
            By default, results are saved in `self._psf_coeff_mod` dictionary. 
            If return_results=True, results are instead returned as function outputs 
            and will not be saved to the dictionary attributes. 
        return_raw : bool
            Normally, we return the relation between PSF coefficients as a function
            of position. Instead this returns (as function outputs) the raw values
            prior to fitting. Final results will not be saved to the dictionary attributes.
        """
        return _gen_wfefield_coeff(self, force=force, save=save, **kwargs)


    def calc_psf_from_coeff(self, sp=None, return_oversample=True, wfe_drift=None, 
        coord_vals=None, coord_frame='tel', coron_rescale=False, return_hdul=True, 
        **kwargs):
        """ Create PSF image from polynomial coefficients
        
        Create a PSF image from instrument settings. The image is noiseless and
        doesn't take into account any non-linearity or saturation effects, but is
        convolved with the instrument throughput. Pixel values are in counts/sec.
        The result is effectively an idealized slope image (no background).

        Returns a single image or list of images if sp is a list of spectra. 
        By default, it returns only the oversampled PSF, but setting 
        return_oversample=False will instead return detector-sampled images.

        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            If not specified, the default is flat in phot lam 
            (equal number of photons per spectral bin).
            The default is normalized to produce 1 count/sec within that bandpass,
            assuming the telescope collecting area and instrument bandpass. 
            Coronagraphic PSFs will further decrease this due to the smaller pupil
            size and coronagraphic spot. 
        return_oversample : bool
            Returns the oversampled version of the PSF instead of detector-sampled PSF.
            Default: True.
        wfe_drift : float or None
            Wavefront error drift amplitude in nm.
        coord_vals : tuple or None
            Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
            If multiple values, then this should be an array ([xvals], [yvals]).
        coord_frame : str
            Type of input coordinates. 

                * 'tel': arcsecs V2,V3
                * 'sci': pixels, in DMS axes orientation; aperture-dependent
                * 'det': pixels, in raw detector read out axes orientation
                * 'idl': arcsecs relative to aperture reference location.

        return_hdul : bool
            Return PSFs in an HDUList rather than set of arrays (default: True).
        coron_rescale : bool
            Rescale off-axis coronagraphic PSF to better match analytic prediction
            when source overlaps coronagraphic occulting mask.
            Primarily used for planetary companion PSFs.
        """        

        res = _calc_psf_from_coeff(self, sp=sp, return_oversample=return_oversample, 
                                   coord_vals=coord_vals, coord_frame=coord_frame, 
                                   wfe_drift=wfe_drift, return_hdul=return_hdul, **kwargs)

        # Ensure correct scaling for off-axis PSFs
        if self.is_coron and coron_rescale and (coord_vals is not None):
            siaf_ap = kwargs.get('siaf_ap', None)
            res = _nrc_coron_rescale(self, res, coord_vals, coord_frame, siaf_ap=siaf_ap, sp=sp)

        return res


    def calc_psf(self, add_distortion=None, fov_pixels=None, oversample=None, 
        wfe_drift=None, coord_vals=None, coord_frame='tel', **kwargs):
        """ Compute a PSF

        Slight modification of inherent WebbPSF `calc_psf` function. If add_distortion, fov_pixels,
        and oversample are not specified, then we automatically use the associated attributes. 
        Also, add ability to directly specify wfe_drift and coordinate offset values in the same
        fashion as `calc_psf_from_coeff`.

        Notes
        -----
        Additional PSF computation options (pupil shifts, source positions, jitter, ...)
        may be set by configuring the `.options` dictionary attribute of this class.

        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            Source input spectrum. If not specified, the default is flat in phot lam.
            (equal number of photons per spectral bin).
        source : synphot.spectrum.SourceSpectrum or dict
            TODO: synphot not yet implemented in webbpsf_ext!!
        nlambda : int
            How many wavelengths to model for broadband?
            The default depends on how wide the filter is: (5,3,1) for types (W,M,N) respectively
        monochromatic : float, optional
            Setting this to a wavelength value (in meters) will compute a monochromatic PSF at that
            wavelength, overriding filter and nlambda settings.
        fov_arcsec : float
            field of view in arcsec. Default=5
        fov_pixels : int
            field of view in pixels. This is an alternative to fov_arcsec.
        outfile : string
            Filename to write. If None, then result is returned as an HDUList
        oversample, detector_oversample, fft_oversample : int
            How much to oversample. Default=4. By default the same factor is used for final output
            pixels and intermediate optical planes, but you may optionally use different factors
            if so desired.
        overwrite : bool
            overwrite output FITS file if it already exists?
        display : bool
            Whether to display the PSF when done or not.
        save_intermediates, return_intermediates : bool
            Options for saving to disk or returning to the calling function the intermediate optical planes during
            the propagation. This is useful if you want to e.g. examine the intensity in the Lyot plane for a
            coronagraphic propagation.
        normalize : string
            Desired normalization for output PSFs. See doc string for OpticalSystem.calc_psf. Default is
            to normalize the entrance pupil to have integrated total intensity = 1.
        add_distortion : bool
            If True, will add 2 new extensions to the PSF HDUlist object. The 2nd extension
            will be a distorted version of the over-sampled PSF and the 3rd extension will
            be a distorted version of the detector-sampled PSF.
        crop_psf : bool
            If True, when the PSF is rotated to match the detector's rotation in the focal
            plane, the PSF will be cropped so the shape of the distorted PSF will match it's
            undistorted counterpart. This will only be used for NIRCam, NIRISS, and FGS PSFs.

        Keyword Args
        ------------
        return_hdul : bool
            Return PSFs in an HDUList rather than set of arrays (default: True).
        return_oversample : bool
            Returns the oversampled version of the PSF instead of detector-sampled PSF.
            Only valid for `reaturn_hdul=False`, otherwise full HDUList returned. Default: True.
        """

        calc_psf_func = super().calc_psf
        res = _calc_psf_webbpsf(self, calc_psf_func, add_distortion=add_distortion, 
                                fov_pixels=fov_pixels, oversample=oversample, wfe_drift=wfe_drift, 
                                coord_vals=coord_vals, coord_frame=coord_frame, **kwargs)

        return res

    def calc_psfs_grid(self, sp=None, wfe_drift=0, osamp=1, npsf_per_full_fov=15,
                       xsci_vals=None, ysci_vals=None, return_coords=None, 
                       use_coeff=True, **kwargs):

        """ PSF grid across an instrumnet FoV
        
        Create a grid of PSFs across instrument aperture FoV. By default,
        imaging observations will be for full detector FoV with regularly
        spaced grid. Coronagraphic observations will cover nominal 
        coronagraphic mask region (usually 10s of arcsec) and will have
        logarithmically spaced values where appropriate.

        Keyword Args
        ============
        sp : :mod:`pysynphot.spectrum`
            If not specified, the default is flat in phot lam (equal number of photons 
            per wavelength bin). The default is normalized to produce 1 count/sec within 
            that bandpass, assuming the telescope collecting area and instrument bandpass. 
            Coronagraphic PSFs will further decrease this due to the smaller pupil
            size and suppression of coronagraphic mask. 
            If set, then the resulting PSF image will be scaled to generate the total
            observed number of photons from the spectrum (ie., not scaled by unit response).
        wfe_drift : float
            Desired WFE drift value relative to default OPD.
        osamp : int
            Sampling of output PSF relative to detector sampling.
        npsf_per_full_fov : int
            Number of PSFs across one dimension of the instrument's field of 
            view. If a coronagraphic observation, then this is for the nominal
            coronagrahic field of view (20"x20"). 
        xsci_vals: None or ndarray
            Option to pass a custom grid values along x-axis in 'sci' coords.
        ysci_vals: None or ndarray
            Option to pass a custom grid values along y-axis in 'sci' coords.
        return_coords : None or str
            Option to also return coordinate values in desired frame 
            ('det', 'sci', 'tel', 'idl'). Output is then xvals, yvals, hdul_psfs.
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.
        coron_rescale : bool
            Rescale off-axis coronagraphic PSF to better match analytic prediction
            when source overlaps coronagraphic occulting mask. Only valid for use_coeff=True.
        """

        res = _calc_psfs_grid(self, sp=sp, wfe_drift=wfe_drift, osamp=osamp, npsf_per_full_fov=npsf_per_full_fov,
                              xsci_vals=xsci_vals, ysci_vals=ysci_vals, return_coords=return_coords, 
                              use_coeff=use_coeff, **kwargs)
        return res

    def calc_psfs_sgd(self, xoff_asec, yoff_asec, use_coeff=True, **kwargs):
        """ Calculate small grid dither PSFs

        Convenience function to calculation a series of SGD PSFs. This is
        essentially a wrapper around the `calc_psf_from_coeff` and `calc_psf`
        functions. Only valid for coronagraphic observations.

        Parameters
        ==========
        xoff_asec : float or array-like
            Offsets in x-direction (in 'idl' coordinates).
        yoff_asec : float or array-like
            Offsets in y-direction (in 'idl' coordinates).
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.
        """

        res = _calc_psfs_sgd(self, xoff_asec, yoff_asec, use_coeff=use_coeff, **kwargs)
        return res

# MIRI Subclass
class MIRI_ext(webbpsf_MIRI):
    
    """ MIRI instrument PSF coefficients
    
    Subclass of WebbPSF's MIRI class for generating polynomial coefficients
    to cache and quickly generate PSFs for arbitrary spectral types as well
    as WFE variations due to field-dependent OPDs and telescope thermal drifts.

    Parameters
    ==========
    filter : str
        Name of input filter.
    pupil_mask : str, None
        Pupil elements such as grisms or lyot stops (default: None).
    image_mask : str, None
        Specify which coronagraphic occulter (default: None).
    fov_pix : int
        Size of the PSF FoV in pixels (real SW or LW pixels).
        The defaults depend on the type of observation.
        Odd number place the PSF on the center of the pixel,
        whereas an even number centers it on the "crosshairs."
    oversample : int
        Factor to oversample during WebbPSF calculations.
        Default 2 for coronagraphy and 4 otherwise.
    """

    def __init__(self, filter=None, pupil_mask=None, image_mask=None, 
                 fov_pix=None, oversample=None, **kwargs):
        
        webbpsf_MIRI.__init__(self)
        _init_inst(self, filter=filter, pupil_mask=pupil_mask, image_mask=image_mask,
                   fov_pix=fov_pix, oversample=oversample, **kwargs)

        # No jitter for coronagraphy by default
        # Otherwise assume 5 mas / axis
        self.options['jitter'] = None if self.is_coron else 'gaussian'
        self.options['jitter_sigma'] = 0.005

    @property
    def save_dir(self):
        """Coefficient save directory"""
        return _gen_save_dir(self)
    @save_dir.setter
    def save_dir(self, value):
        self._save_dir = value
    def _erase_save_dir(self):
        """Erase all instrument coefficients"""
        _clear_coeffs_dir(self)

    @property
    def save_name(self):
        """Coefficient file name"""
        if self._save_name is None:
            return self.gen_save_name()
        else:
            return self._save_name
    @save_name.setter
    def save_name(self, value):
        self._save_name = value
                
    @property
    def is_coron(self):
        """
        Coronagraphic observations based on pupil mask settings
        """
        pupil = self.pupil_mask
        return (pupil is not None) and (('LYOT' in pupil) or ('FQPM' in pupil))
    @property
    def is_slitspec(self):
        """
        LRS observations based on pupil mask settings
        """
        pupil = self.pupil_mask
        return (pupil is not None) and ('LRS' in pupil)
    
    @property
    def fov_pix(self):
        return self._fov_pix
    @fov_pix.setter
    def fov_pix(self, value):
        self._fov_pix = value
        
    @property
    def oversample(self):
        if self._oversample is None:
            oversample = 2 if self.is_coron else 4
        else:
            oversample = self._oversample
        return oversample
    @oversample.setter
    def oversample(self, value):
        self._oversample = value
    
    @property
    def wave_fit(self):
        """Wavelength range to fit"""
        if self.quick:
            w1 = self.bandpass.wave.min() / 1e4
            w2 = self.bandpass.wave.max() / 1e4
        else:
            w1, w2 = (5,30)
        return (w1, w2)
    @property
    def npsf(self):
        """Number of wavelengths/PSFs to fit"""

        w1, w2 = self.wave_fit

        npsf = self._npsf
        # Default to 10 PSF simulations per um
        if npsf is None:
            dn = 10 
            npsf = int(np.ceil(dn * (w2-w1)))

        # Want at least 5 monochromatic PSFs
        npsf = 5 if npsf<5 else int(npsf)

        # Number of points must be greater than degree of fit
        npsf = self.ndeg+1 if npsf<=self.ndeg else int(npsf)

        return npsf
    @npsf.setter
    def npsf(self, value):
        """Set number of wavelengths/PSFs to fit"""
        self._npsf = value
        
    @property
    def ndeg(self):
        """Degree of polynomial fit"""
        ndeg = self._ndeg
        if ndeg is None:
            # TODO: Quantify these better
            if self.use_legendre:
                ndeg = 4 if self.quick else 7
            else:
                ndeg = 4 if self.quick else 7
        return ndeg
    @ndeg.setter
    def ndeg(self, value):
        """Degree of polynomial fit"""
        self._ndeg = value

    @property
    def quick(self):
        """Perform quicker coeff calculation over limited bandwidth?"""
        if self._quick is not None:
            quick = self._quick
        else:
            quick = True
        return quick
    @quick.setter
    def quick(self, value):
        """Perform quicker coeff calculation over limited bandwidth?"""
        _check_list(value, [True, False], 'quick')
        self._quick = value

    @property
    def siaf_ap(self):
        """SIAF Aperture object"""
        if self._siaf_ap is None:
            return self.siaf[self.aperturename]
        else:
            return self._siaf_ap
    @siaf_ap.setter
    def siaf_ap(self, value):
        self._siaf_ap = value

    @webbpsf_MIRI.detector_position.setter
    def detector_position(self, position):
        try:
            x, y = map(float, position)
        except ValueError:
            raise ValueError("Detector pixel coordinates must be a pair of numbers, not {}".format(position))
        self._detector_position = (x,y)

    def _get_fits_header(self, result, options):
        """ populate FITS Header keywords """
        super(MIRI_ext, self)._get_fits_header(result, options)

        # Keep detector X and Y positions as floats
        dpos = np.asarray(self.detector_position, dtype=float)
        result[0].header['DET_X'] = (dpos[0], "Detector X pixel position of array center")
        result[0].header['DET_Y'] = (dpos[1], "Detector Y pixel position of array center")
    
    @property
    def fastaxis(self):
        """Fast readout direction in sci coords"""
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#orientation-of-detector-image
        # MIRI always has fastaxis equal +1
        return +1
    @property
    def slowaxis(self):
        """Slow readout direction in sci coords"""
        # https://jwst-pipeline.readthedocs.io/en/latest/jwst/references_general/references_general.html#orientation-of-detector-image
        # MIRI always has slowaxis equal +2
        return +2

    @property
    def bandpass(self):
        return miri_filter(self.filter)

    def gen_mask_image(self, npix=None, pixelscale=None, detector_orientation=True):
        """
        Return an image representation of the focal plane mask.
        For 4QPM, we should the phase offsets (0 or 1), whereas
        the Lyot and LRS slit masks return transmission.
        
        Parameters
        ==========
        npix : int
            Number of pixels in output image. If not set, then
            is automatically determined based on mask FoV and
            `pixelscale`
        pixelscale : float
            Size of output pixels in units of arcsec. If not specified,
            then selects nominal detector pixel scale.
        detector_orientation : bool
            Should the output image be rotated to be in detector coordinates?
            If set to False, then output mask is rotated along V2/V3 axes.
        """
        
        def make_fqpm_wrapper(name, wavelength):
            opticslist = [poppy.IdealFQPM(wavelength=wavelength, name=self.image_mask, rotation=rot1, **offsets),
                          poppy.SquareFieldStop(size=24, rotation=rot2, **offsets)]
            container = poppy.CompoundAnalyticOptic(name=name, opticslist=opticslist)
            return container
        
        rot1 = -1*self._rotation if detector_orientation else 0
        rot2 = 0 if detector_orientation else self._rotation
        offsets = {'shift_x': self.options.get('coron_shift_x', None),
                   'shift_y': self.options.get('coron_shift_y', None)}
        
        if pixelscale is None:
            pixelscale = self.pixelscale / self.oversample

        if self.image_mask == 'FQPM1065':
            full_pad = 2*np.max(np.abs(xy_rot(12, 12, rot2)))
            npix = int(full_pad / pixelscale + 0.5) if npix is None else npix
            wave = poppy.Wavefront(wavelength=10.65e-6, npix=npix, pixelscale=pixelscale)
            mask = make_fqpm_wrapper("MIRI FQPM 1065", 10.65e-6)
            im = np.real(mask.get_phasor(wave))
            im /= im.max()
        elif self.image_mask == 'FQPM1140':
            full_pad = 2*np.max(np.abs(xy_rot(12, 12, rot2)))
            npix = int(full_pad / pixelscale + 0.5) if npix is None else npix
            wave = poppy.Wavefront(wavelength=11.4e-6, npix=npix, pixelscale=pixelscale)
            mask = make_fqpm_wrapper("MIRI FQPM 1140", 11.40e-6)
            im = np.real(mask.get_phasor(wave))
            im /= im.max()
        elif self.image_mask == 'FQPM1550':
            full_pad = 2*np.max(np.abs(xy_rot(12, 12, rot2)))
            npix = int(full_pad / pixelscale + 0.5) if npix is None else npix
            wave = poppy.Wavefront(wavelength=15.5e-6, npix=npix, pixelscale=pixelscale)
            mask = make_fqpm_wrapper("MIRI FQPM 1550", 15.50e-6)
            im = np.real(mask.get_phasor(wave))
            im /= im.max()
        elif self.image_mask == 'LYOT2300':
            full_pad = 2*np.max(np.abs(xy_rot(15, 15, rot2)))
            npix = int(full_pad / pixelscale + 0.5) if npix is None else npix
            wave = poppy.Wavefront(wavelength=23e-6, npix=npix, pixelscale=pixelscale)
            opticslist = [poppy.CircularOcculter(radius=4.25 / 2, name=self.image_mask, rotation=rot1, **offsets),
                          poppy.BarOcculter(width=0.722, height=31, rotation=rot1, **offsets),
                          poppy.SquareFieldStop(size=30, rotation=rot2, **offsets)]
            mask = poppy.CompoundAnalyticOptic(name="MIRI Lyot Occulter", opticslist=opticslist)
            im = mask.get_transmission(wave)**2
        elif self.image_mask == 'LRS slit':
            full_pad = 2*np.max(np.abs(xy_rot(2.5, 2.5, rot2)))
            npix = int(full_pad / pixelscale + 0.5) if npix is None else npix
            wave = poppy.Wavefront(wavelength=23e-6, npix=npix, pixelscale=pixelscale)
            mask = poppy.RectangularFieldStop(width=4.7, height=0.51, rotation=rot2, 
                                              name=self.image_mask, **offsets)
            im = mask.get_transmission(wave)**2
        else:
            im = np.ones([npix,npix])
        
        return im
        
    def get_opd_info(self, opd=None, pupil=None, HDUL_to_OTELM=True):
        """
        Parse out OPD information for a given OPD, which 
        can be a file name, tuple (file,slice), HDUList,
        or OTE Linear Model. Returns dictionary of some
        relevant information for logging purposes.
        The dictionary has an OPD version as an OTE LM.
        
        This outputs an OTE Linear Model. 
        In order to update instrument class:

            >>> opd_dict = inst.get_opd_info()
            >>> opd_new = opd_dict['pupilopd']
            >>> inst.pupilopd = opd_new
            >>> inst.pupil = opd_new
        """
        return _get_opd_info(self, opd=opd, pupil=pupil, HDUL_to_OTELM=HDUL_to_OTELM)
    
    def drift_opd(self, wfe_drift, opd=None):
        """
        A quick method to drift the pupil OPD. This function applies 
        some WFE drift to input OPD file by breaking up the wfe_drift 
        attribute into thermal, frill, and IEC components. If we want 
        more realistic time evolution, then we should use the procedure 
        in dev_utils/WebbPSF_OTE_LM.ipynb to create a time series of OPD
        maps, which can then be passed directly to create unique PSFs.
        
        This outputs an OTE Linear Model. In order to update instrument class:
        
            >>> opd_dict = inst.drift_opd()
            >>> inst.pupilopd = opd_dict['opd']
            >>> inst.pupil = opd_dict['opd']
        """
        return _drift_opd(self, wfe_drift, opd=opd)

    def gen_save_name(self, wfe_drift=0):
        """
        Generate save name for polynomial coefficient output file.
        """
        return _gen_save_name(self, wfe_drift=wfe_drift)

    def gen_psf_coeff(self, **kwargs):
        """Generate PSF coefficients

        Creates a set of coefficients that will generate simulated PSFs for any
        arbitrary wavelength. This function first simulates a number of evenly-
        spaced PSFs throughout the specified bandpass (or the full channel). 
        An nth-degree polynomial is then fit to each oversampled pixel using 
        a linear-least squares fitting routine. The final set of coefficients 
        for each pixel is returned as an image cube. The returned set of 
        coefficient are then used to produce PSF via `calc_psf_from_coeff`.

        Useful for quickly generated imaging and dispersed PSFs for multiple
        spectral types. 

        Keyword Args
        ------------
        wfe_drift : float
            Wavefront error drift amplitude in nm.
        force : bool
            Forces a recalculation of PSF even if saved PSF exists. (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)
        nproc : bool or None
            Manual setting of number of processor cores to break up PSF calculation.
            If set to None, this is determined based on the requested PSF size,
            number of available memory, and hardware processor cores. The automatic
            calculation endeavors to leave a number of resources available to the
            user so as to not crash the user's machine. 
        return_results : bool
            By default, results are saved as object the attributes `psf_coeff` and
            `psf_coeff_header`. If return_results=True, results are instead returned
            as function outputs and will not be saved to the attributes. This is mostly
            used for successive coeff simulations to determine varying WFE drift or 
            focal plane dependencies.
        return_extras : bool
            Additionally returns a dictionary of monochromatic PSFs images and their 
            corresponding wavelengths for debugging purposes. Can be used with or without
            `return_results`. If `return_results=False`, then only this dictionary is
            returned, otherwise if `return_results=False` then returns everything as a
            3-element tuple (psf_coeff, psf_coeff_header, extras_dict).
        """
        
        return _gen_psf_coeff(self, **kwargs)

    def gen_wfedrift_coeff(self, force=False, save=True, **kwargs):
        """ Fit WFE drift coefficients

        This function finds a relationship between PSF coefficients in the 
        presence of WFE drift. For a series of WFE drift values, we generate 
        corresponding PSF coefficients and fit a  polynomial relationship to 
        the residual values. This allows us to quickly modify a nominal set of 
        PSF image coefficients to generate a new PSF where the WFE has drifted 
        by some amplitude.
        
        It's Legendre's all the way down...

        Parameters
        ----------
        force : bool
            Forces a recalculation of coefficients even if saved file exists. 
            (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)

        Keyword Args
        ------------
        wfe_list : array-like
            A list of wavefront error drift values (nm) to calculate and fit.
            Default is [0,1,2,5,10,20,40], which covers the most-likely
            scenarios (1-5nm) while also covering a range of extreme drift
            values (10-40nm).
        return_results : bool
            By default, results are saved in `self._psf_coeff_mod` dictionary. 
            If return_results=True, results are instead returned as function outputs 
            and will not be saved to the dictionary attributes. 
        return_raw : bool
            Normally, we return the relation between PSF coefficients as a function
            of position. Instead this returns (as function outputs) the raw values
            prior to fitting. Final results will not be saved to the dictionary attributes.
        """
        return _gen_wfedrift_coeff(self, force=force, save=save, **kwargs)

    def gen_wfefield_coeff(self, force=False, save=True, **kwargs):
        """ Fit WFE field-dependent coefficients

        Find a relationship between field position and PSF coefficients for
        non-coronagraphic observations and when `include_si_wfe` is enabled.

        Parameters
        ----------
        force : bool
            Forces a recalculation of coefficients even if saved file exists. 
            (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)

        Keyword Args
        ------------
        return_results : bool
            By default, results are saved in `self._psf_coeff_mod` dictionary. 
            If return_results=True, results are instead returned as function outputs 
            and will not be saved to the dictionary attributes. 
        return_raw : bool
            Normally, we return the relation between PSF coefficients as a function
            of position. Instead this returns (as function outputs) the raw values
            prior to fitting. Final results will not be saved to the dictionary attributes.
        """
        return _gen_wfefield_coeff(self, force=force, save=save, **kwargs)

    def gen_wfemask_coeff(self, large_grid=False, force=False, save=True, **kwargs):
        """ Fit WFE changes in mask position

        For coronagraphic masks, slight changes in the PSF location
        relative to the image plane mask can substantially alter the 
        PSF speckle pattern. This function generates a number of PSF
        coefficients at a variety of positions, then fits polynomials
        to the residuals to track how the PSF changes across the mask's
        field of view. Special care is taken near the 10-20mas region
        in order to provide accurate sampling of the SGD offsets. 

        Parameters
        ----------
        large_grid : bool
            Use a large number (high-density) of grid points to create coefficients.
            If True, then a higher fidelity PSF variations across the FoV, but could
            take hours to genrate on the first pass. Setting to False allows for
            quicker coefficient creation with a smaller memory footprint, useful for
            testing and debugging.        
        force : bool
            Forces a recalculation of coefficients even if saved file exists. 
            (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)

        Keyword Args
        ------------
        return_results : bool
            By default, results are saved in `self._psf_coeff_mod` dictionary. 
            If return_results=True, results are instead returned as function outputs 
            and will not be saved to the dictionary attributes. 
        return_raw : bool
            Normally, we return the relation between PSF coefficients as a function
            of position. Instead this returns (as function outputs) the raw values
            prior to fitting. Final results will not be saved to the dictionary attributes.

        """
        return _gen_wfemask_coeff(self, large_grid=large_grid, force=force, save=save, **kwargs)

    def calc_psf_from_coeff(self, sp=None, return_oversample=True, return_hdul=True,
        wfe_drift=None, coord_vals=None, coord_frame='tel', **kwargs):
        """ Create PSF image from coefficients
        
        Create a PSF image from instrument settings. The image is noiseless and
        doesn't take into account any non-linearity or saturation effects, but is
        convolved with the instrument throughput. Pixel values are in counts/sec.
        The result is effectively an idealized slope image (no background).

        Returns a single image or list of images if sp is a list of spectra. 
        By default, it returns only the oversampled PSF, but setting 
        return_oversample=False will instead return detector-sampled images.

        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            If not specified, the default is flat in phot lam 
            (equal number of photons per spectral bin).
            The default is normalized to produce 1 count/sec within that bandpass,
            assuming the telescope collecting area and instrument bandpass. 
            Coronagraphic PSFs will further decrease this due to the smaller pupil
            size and coronagraphic mask. 
        return_oversample : bool
            Returns the oversampled version of the PSF instead of detector-sampled PSF.
            Default: True.
        wfe_drift : float or None
            Wavefront error drift amplitude in nm.
        coord_vals : tuple or None
            Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
            If multiple values, then this should be an array ([xvals], [yvals]).
        coord_frame : str
            Type of input coordinates. 

                * 'tel': arcsecs V2,V3
                * 'sci': pixels, in conventional DMS axes orientation
                * 'det': pixels, in raw detector read out axes orientation
                * 'idl': arcsecs relative to aperture reference location.

        return_hdul : bool
            Return PSFs in an HDUList rather than set of arrays (default: True).
        """        
        return _calc_psf_from_coeff(self, sp=sp, return_oversample=return_oversample, 
                                    coord_vals=coord_vals, coord_frame=coord_frame, 
                                    wfe_drift=wfe_drift, return_hdul=return_hdul, **kwargs)

    def calc_psf(self, add_distortion=None, fov_pixels=None, oversample=None, 
        wfe_drift=None, coord_vals=None, coord_frame='tel', **kwargs):
        """ Compute a PSF

        Slight modification of inherent WebbPSF `calc_psf` function. If add_distortion, fov_pixels,
        and oversample are not specified, then we automatically use the associated attributes.

        Notes
        -----
        More advanced PSF computation options (pupil shifts, source positions, jitter, ...)
        may be set by configuring the `.options` dictionary attribute of this class.

        Parameters
        ----------
        sp : :mod:`pysynphot.spectrum`
            Source input spectrum. If not specified, the default is flat in phot lam.
            (equal number of photons per spectral bin).
        source : synphot.spectrum.SourceSpectrum or dict
            TODO: synphot not yet implemented in webbpsf_ext!!
        nlambda : int
            How many wavelengths to model for broadband?
            The default depends on how wide the filter is: (5,3,1) for types (W,M,N) respectively
        monochromatic : float, optional
            Setting this to a wavelength value (in meters) will compute a monochromatic PSF at that
            wavelength, overriding filter and nlambda settings.
        fov_arcsec : float
            field of view in arcsec. Default=5
        fov_pixels : int
            field of view in pixels. This is an alternative to fov_arcsec.
        outfile : string
            Filename to write. If None, then result is returned as an HDUList
        oversample, detector_oversample, fft_oversample : int
            How much to oversample. Default=4. By default the same factor is used for final output
            pixels and intermediate optical planes, but you may optionally use different factors
            if so desired.
        overwrite : bool
            overwrite output FITS file if it already exists?
        display : bool
            Whether to display the PSF when done or not.
        save_intermediates, return_intermediates : bool
            Options for saving to disk or returning to the calling function the intermediate optical planes during
            the propagation. This is useful if you want to e.g. examine the intensity in the Lyot plane for a
            coronagraphic propagation.
        normalize : string
            Desired normalization for output PSFs. See doc string for OpticalSystem.calc_psf. Default is
            to normalize the entrance pupil to have integrated total intensity = 1.
        add_distortion : bool
            If True, will add 2 new extensions to the PSF HDUlist object. The 2nd extension
            will be a distorted version of the over-sampled PSF and the 3rd extension will
            be a distorted version of the detector-sampled PSF.
        crop_psf : bool
            If True, when the PSF is rotated to match the detector's rotation in the focal
            plane, the PSF will be cropped so the shape of the distorted PSF will match it's
            undistorted counterpart. This will only be used for NIRCam, NIRISS, and FGS PSFs.

        Keyword Args
        ------------
        return_hdul : bool
            Return PSFs in an HDUList rather than set of arrays (default: True).
        return_oversample : bool
            Returns the oversampled version of the PSF instead of detector-sampled PSF.
            Only valid for `reaturn_hdul=False`, otherwise full HDUList returned. Default: True.
        """
        calc_psf_func = super().calc_psf
        res = _calc_psf_webbpsf(self, calc_psf_func, add_distortion=add_distortion, 
                                fov_pixels=fov_pixels, oversample=oversample, wfe_drift=wfe_drift, 
                                coord_vals=coord_vals, coord_frame=coord_frame, **kwargs)

        return res

    def calc_psfs_grid(self, sp=None, wfe_drift=0, osamp=1, npsf_per_full_fov=15,
                       xsci_vals=None, ysci_vals=None, return_coords=None, 
                       use_coeff=True, **kwargs):

        """ PSF grid across an instrumnet FoV
        
        Create a grid of PSFs across instrument aperture FoV. By default,
        imaging observations will be for full detector FoV with regularly
        spaced grid. Coronagraphic observations will cover nominal 
        coronagraphic mask region (usually 10s of arcsec) and will have
        logarithmically spaced values where appropriate.

        Keyword Args
        ============
        sp : :mod:`pysynphot.spectrum`
            If not specified, the default is flat in phot lam (equal number of photons 
            per wavelength bin). The default is normalized to produce 1 count/sec within 
            that bandpass, assuming the telescope collecting area and instrument bandpass. 
            Coronagraphic PSFs will further decrease this due to the smaller pupil
            size and suppression of coronagraphic mask. 
            If set, then the resulting PSF image will be scaled to generate the total
            observed number of photons from the spectrum (ie., not scaled by unit response).     
        wfe_drift : float
            Desired WFE drift value relative to default OPD.
        osamp : int
            Sampling of output PSF relative to detector sampling.
        npsf_per_full_fov : int
            Number of PSFs across one dimension of the instrument's field of 
            view. If a coronagraphic observation, then this is for the nominal
            coronagrahic field of view. 
        xsci_vals: None or ndarray
            Option to pass a custom grid values along x-axis in 'sci' coords.
            If coronagraph, this instead corresponds to coronagraphic mask axis, 
            which has a slight rotation relative to detector axis in MIRI.
        ysci_vals: None or ndarray
            Option to pass a custom grid values along y-axis in 'sci' coords.
            If coronagraph, this instead corresponds to coronagraphic mask axis, 
            which has a slight rotation relative to detector axis in MIRI.
        return_coords : None or str
            Option to also return coordinate values in desired frame 
            ('det', 'sci', 'tel', 'idl'). Output is then xvals, yvals, hdul_psfs.
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.
        """

        res = _calc_psfs_grid(self, sp=sp, wfe_drift=wfe_drift, osamp=osamp, npsf_per_full_fov=npsf_per_full_fov,
                              xsci_vals=xsci_vals, ysci_vals=ysci_vals, return_coords=return_coords, 
                              use_coeff=use_coeff, **kwargs)
        return res

    def calc_psfs_sgd(self, xoff_asec, yoff_asec, use_coeff=True, **kwargs):
        """ Calculate small grid dither PSFs

        Convenience function to calculation a series of SGD PSFs. This is
        essentially a wrapper around the `calc_psf_from_coeff` and `calc_psf`
        functions. Only valid for coronagraphic observations.

        Parameters
        ==========
        xoff_asec : float or array-like
            Offsets in x-direction (in 'idl' coordinates).
        yoff_asec : float or array-like
            Offsets in y-direction (in 'idl' coordinates).
        use_coeff : bool
            If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.
        """

        res = _calc_psfs_sgd(self, xoff_asec, yoff_asec, use_coeff=use_coeff, **kwargs)
        return res

#############################################################
#  Functions for use across instrument classes
#############################################################

def _check_list(value, temp_list, var_name=None):
    """
    Helper function to test if a value exists within a list. 
    If not, then raise ValueError exception.
    This is mainly used for limiting the allowed values of some variable.
    """
    if value not in temp_list:
        # Replace None value with string for printing
        if None in temp_list: 
            temp_list[temp_list.index(None)] = 'None'
        # Make sure all elements are strings
        temp_list2 = [str(val) for val in temp_list]
        var_name = '' if var_name is None else var_name + ' '
        err_str = "Invalid {}setting: {} \n\tValid values are: {}" \
                         .format(var_name, value, ', '.join(temp_list2))
        raise ValueError(err_str)


def _init_inst(self, filter=None, pupil_mask=None, image_mask=None, 
               fov_pix=None, oversample=None, **kwargs):
    """
    Setup for specific instrument during init state
    """

    # Add grisms as pupil options
    if self.name=='NIRCam':
        self.pupil_mask_list = self.pupil_mask_list + ['GRISMC', 'GRISMR', 'FLAT']
    elif self.name=='NIRISS':
        self.pupil_mask_list = self.pupil_mask_list + ['GR150C', 'GR150R']

    # Option to use `pupil` instead of `pupil_mask`
    kw_pupil = kwargs.get('pupil')
    kw_mask  = kwargs.get('mask')
    if (pupil_mask is None) and (kw_pupil is not None) and isinstance(kw_pupil, str):
        _log.warn('Deprecation warning: Use `pupil_mask` keyword rather than `pupil` to set pupil mask.')
        # pupil_mask = kwargs.get('pupil')
    # Option to use `mask` instead of `image_mask`
    if (image_mask is None) and (kw_mask is not None):
        _log.warn('Deprecation warning: Use `image_mask` keyword rather than `mask` to set image mask.')
        # image_mask = kwargs.get('mask')

    # Ensure CIRCLYOT or WEDGELYOT in case coron masks were specified
    if self.name=='NIRCam' and (pupil_mask is not None) and ('MASK' in pupil_mask):
        pupil_mask = 'WEDGELYOT' if ('LWB' in pupil_mask) else 'CIRCLYOT'

    if pupil_mask is not None:
        self.pupil_mask = pupil_mask
    if image_mask is not None:
        self.image_mask = image_mask
    # Do filter last
    if filter is not None:
        self.filter = filter
        
    # Don't include SI WFE error for coronagraphy
    if self.name=='MIRI':
        self.include_si_wfe = False if self.is_coron else True
    elif self.name=='NIRCam':
        self.include_si_wfe = True
    else:
        self.include_si_wfe = True

    # SIAF aperture attribute
    self._siaf_ap = None

    # Options to include or exclude distortions
    self.include_distortions = True
    
    # Settings for fov_pix and oversample
    # Default odd for normal imaging, even for coronagraphy
    if fov_pix is None:
        # TODO: Do these even/odd settings make sense?
        # fov_pix = 128 if self.is_coron else 129
        fov_pix = 257
    self._fov_pix = fov_pix
    self._oversample = oversample

    # Legendre polynomials are more stable
    # self.use_legendre = True
    self.use_legendre = kwargs.get('use_legendre', True)    

    # Turning on quick perform fits over filter bandpasses independently
    # The smaller wavelength range requires fewer monochromaic wavelengths
    # and lower order polynomial fits
    self._quick = None
    self.quick = kwargs.get('quick', self.quick)

    # Setting these to None to choose default values at runtime
    self._npsf = None
    self._ndeg = None
    self.npsf = kwargs.get('npsf', self.npsf)
    self.ndeg = kwargs.get('ndeg', self.ndeg)
    
    # Set up initial OPD file info
    opd_name = 'JWST_OTE_OPD_RevAA_prelaunch_predicted.fits'
    try:
        opd_name = check_fitsgz(opd_name)
    except OSError:
        opd_name = f'OPD_RevW_ote_for_{self.name}_predicted.fits'
        opd_name = check_fitsgz(opd_name, self.name)
    self._opd_default = (opd_name, 0)
    self.pupilopd = self._opd_default

    # Update telescope pupil and pupil OPD
    kw_pupil    = kwargs.get('pupil')
    kw_pupilopd = kwargs.get('pupilopd')
    # if (kw_pupil is not None) or (kw_pupilopd is not None):
    #     opd_dict = self.get_opd_info(opd=kw_pupilopd, pupil=kw_pupil)
    #     otelm = opd_dict['pupilopd']
    #     self.pupilopd = otelm
    #     self.pupil    = otelm
    if kw_pupil is not None:
        self.pupil = kw_pupil
    if kw_pupilopd is not None:
        self.pupilopd = kw_pupilopd


    # Name to save array of oversampled coefficients
    self._save_dir = None
    self._save_name = None

    # Max FoVs for calculating drift and field-dependent coefficient residuals
    # Any pixels beyond this size will be considered to have 0 residual difference
    self._fovmax_wfedrift = 256
    self._fovmax_wfefield = 128
    self._fovmax_wfemask  = 256

    self.psf_coeff = None
    self.psf_coeff_header = None
    self._psf_coeff_mod = {
        'wfe_drift': None, 'wfe_drift_off': None, 'wfe_drift_lxmap': None,
        'si_field': None, 'si_field_v2grid': None, 'si_field_v3grid': None, 'si_field_apname': None,
        'si_mask': None, 'si_mask_xgrid': None, 'si_mask_ygrid': None, 'si_mask_apname': None,
        'si_mask_large': False
    }
    if self.image_mask is not None:
        self.options['coron_shift_x'] = 0
        self.options['coron_shift_y'] = 0

def _gen_save_dir(self):
    """
    Generate a default save directory to store PSF coefficients.
    If the directory doesn't exist, try to create it.
    """
    if self._save_dir is None:
        base_dir = Path(conf.WEBBPSF_EXT_PATH) / 'psf_coeffs/'
        # Name to save array of oversampled coefficients
        inst_name = self.name
        save_dir = base_dir / f'{inst_name}/'
    else:
        save_dir = self._save_dir

    if isinstance(save_dir, str):
        save_dir = Path(save_dir)
        self._save_dir = save_dir

    # Create directory (and all intermediates) if it doesn't already exist
    if not os.path.isdir(save_dir):
        _log.info(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    return save_dir

def _clear_coeffs_dir(self):
    """
    Remove contents of a instrument coefficient directory.
    """

    import shutil

    # Should be a pathlib.Path object
    save_dir = self.save_dir
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    if save_dir.exists() and save_dir.is_dir():
        _log.warn(f"Remove contents from '{save_dir}/'?")
        _log.warn("Type 'Y' to continue...")
        response = input("")
        if response=="Y":
            # Delete directory and contents
            shutil.rmtree(save_dir)
            # Recreate empty directory
            os.makedirs(save_dir, exist_ok=True)
            _log.warn("Directory emptied.")
        else:
            _log.warn("Process aborted.")
    else:
        _log.warn(f"Directory '{save_dir}/' does not exist!")


def _gen_save_name(self, wfe_drift=0):
    """
    Create save name for polynomial coefficients output file.
    """
    
    # Prepend filter name if using quick keyword
    fstr = '{}_'.format(self.filter) if self.quick else ''
    # Mask and pupil names
    pstr = 'CLEAR' if self.pupil_mask is None else self.pupil_mask
    mstr = 'NONE' if self.image_mask is None else self.image_mask
    ## 9/14/2022 - PSF weighting for substrate and ND mask should not be necessary
    ## since these are included in bandpass throughputs, which are then
    ## applied to input spectrum to get flux-dependent PSFs. Therefore, the
    ## saved PSF coefficients are similar for all three scenario:
    ##   1) coron_substrate=False; 2) coron_substrate=True; 3) ND_acq=True
    # Check for coron substrate if image mask is None
    # if (mstr == 'NONE') and self.coron_substrate:
    #     mstr = 'CORONSUB'
    # Only need coron substrate for PSF weighting
    # if (mstr == 'NONE'):
    #     if self.ND_acq:
    #         mstr = 'NDACQ'
    #     elif self.coron_substrate:
    #         mstr = 'CORONSUB'

    fmp_str = f'{fstr}{pstr}_{mstr}'

    # PSF image size and sampling
    fov_pix = self.fov_pix
    osamp = self.oversample

    if self.name=='NIRCam':
        # Prepend module and channel to filter/pupil/mask
        module = self.module
        chan_str = 'LW' if 'long' in self.channel else 'SW'
        fmp_str = f'{chan_str}{module}_{fmp_str}'
        # Set bar offset if specified
        # bar_offset = self.options.get('bar_offset', None)
        bar_offset = self.get_bar_offset()
        bar_str = '' if bar_offset is None else '_bar{:.2f}'.format(bar_offset)
    else:
        bar_str = ''

    # Jitter settings
    jitter = self.options.get('jitter')
    jitter_sigma = self.options.get('jitter_sigma', 0)
    if (jitter is None) or (jitter_sigma is None):
        jitter_sigma = 0
    jsig_mas = jitter_sigma*1000
    
    # Source positioning
    offset_r = self.options.get('source_offset_r', 0)
    offset_theta = self.options.get('source_offset_theta', 0)
    if offset_r is None: 
        offset_r = 0
    if offset_theta is None: 
        offset_theta = 0
    rth_str = f'r{offset_r:.2f}_th{offset_theta:+.1f}'
    
    # Mask offsetting
    coron_shift_x = self.options.get('coron_shift_x', 0)
    coron_shift_y = self.options.get('coron_shift_y', 0)
    if coron_shift_x is None: 
        coron_shift_x = 0
    if coron_shift_y is None: 
        coron_shift_y = 0
    moff_str1 = '' if coron_shift_x==0 else f'_mx{coron_shift_x:.3f}'
    moff_str2 = '' if coron_shift_y==0 else f'_my{coron_shift_y:.3f}'
    moff_str = moff_str1 + moff_str2
    
    opd_dict = self.get_opd_info()
    opd_str = opd_dict['opd_str']

    if wfe_drift!=0:
        opd_str = '{}-{:.0f}nm'.format(opd_str,wfe_drift)
    
    fname = f'{fmp_str}_pix{fov_pix}_os{osamp}_jsig{jsig_mas:.0f}_{rth_str}{moff_str}{bar_str}_{opd_str}'
    
    # Add SI WFE tag if included
    if self.include_si_wfe:
        fname = fname + '_siwfe'

    # Add distortions tag if included
    if self.include_distortions:
        fname = fname + '_distort'

    if self.use_legendre:
        fname = fname + '_legendre'

    fname = fname + '.fits'
    
    return fname


def _get_opd_info(self, opd=None, pupil=None, HDUL_to_OTELM=True):
    """
    Parse out OPD information for a given OPD, which can be a 
    file name, tuple (file,slice), HDUList, or OTE Linear Model. 
    Returns dictionary of some relevant information for logging purposes.
    The dictionary has an OPD version as an OTE_Linear_Model_WSS object.
    
    This outputs an OTE Linear Model. 
    In order to update instrument class:
        >>> opd_dict = inst.get_opd_info()
        >>> opd_new = opd_dict['pupilopd']
        >>> inst.pupilopd = opd_new
        >>> inst.pupil = opd_new
    """
    
    # Pupil OPD file name
    if opd is None:
        opd = self.pupilopd

    if pupil is None:
        pupil = self.pupil
        
    # If OPD is None or a string, make into tuple
    if opd is None:  # Default OPD
        opd = self._opd_default
    elif isinstance(opd, six.string_types):
        opd = (opd, 0)

    # Change log levels to WARNING 
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    # Parse OPD info
    if isinstance(opd, tuple):
        if not len(opd)==2:
            raise ValueError("opd passed as tuple must have length of 2.")
        # Filename info
        opd_name = opd[0] # OPD file name
        opd_num  = opd[1] # OPD slice
        rev = [s for s in opd_name.split('_') if "Rev" in s]
        rev = '' if len(rev)==0 else rev[0]
        opd_str = '{}slice{:.0f}'.format(rev,opd_num)
        opd = OPDFile_to_HDUList(opd_name, opd_num)
    elif isinstance(opd, fits.HDUList):
        # A custom OPD is passed. 
        opd_name = 'OPD from FITS HDUlist'
        opd_num = 0
        opd_str = 'OPDcustomFITS'
    elif isinstance(opd, poppy.OpticalElement):
        # OTE Linear Model
        # opd_name = 'OPD from OTE LM'
        opd_name = opd.name
        opd_num = 0
        opd_str = 'OPDcustomLM'
    else:
        raise ValueError("OPD must be a string, tuple, HDUList, or OTE LM.")
        
    # OPD should now be an HDUList or OTE LM
    # Convert to OTE LM if HDUList
    if HDUL_to_OTELM and isinstance(opd, fits.HDUList):
        hdul = opd

        header = hdul[0].header
        header['ORIGINAL'] = (opd_name,   "Original OPD source")
        header['SLICE']    = (opd_num,    "Slice index of original OPD")
        #header['WFEDRIFT'] = (self.wfe_drift, "WFE drift amount [nm]")

        name = 'Modified from ' + opd_name
        opd = OTE_Linear_Model_WSS(name=name, transmission=pupil,
                                   opd=hdul, opd_index=opd_num, 
                                   v2v3=self._tel_coords())
        
    setup_logging(log_prev, verbose=False)

    out_dict = {'opd_name':opd_name, 'opd_num':opd_num, 'opd_str':opd_str, 'pupilopd':opd}
    return out_dict

def _drift_opd(self, wfe_drift, opd=None, wfe_therm=None, wfe_frill=None, wfe_iec=None):
    """
    A quick method to drift the pupil OPD. This function applies 
    some WFE drift to input OPD file by breaking up the wfe_drift 
    parameter into thermal, frill, and IEC components. If we want 
    more realistic time evolution, then we should use the procedure 
    in dev_utils/WebbPSF_OTE_LM.ipynb to create a time series of OPD
    maps, which can then be passed directly to create unique PSFs.
    
    This outputs an OTE Linear Model. In order to update instrument class:
        >>> opd_dict = inst.drift_opd()
        >>> inst.pupilopd = opd_dict['opd']
        >>> inst.pupil = opd_dict['opd']

    Parameters
    ----------
    wfe_drift : float
        Desired WFE drift (delta RMS) in nm.
    opd : Various
        file name, tuple (file,slice), HDUList, or OTE Linear Model
        of the OPD map.
    wfe_therm : None or float
        Option to specify thermal component of WFE drift (nm RMS). 
        `wfe_drift` is ignored.
    wfe_frill : None or float
        Option to specify frill component of WFE drift (nm RMS). 
        `wfe_drift` is ignored.
    wfe_iec : None or float
        Option to specify IEC component of WFE drift (nm RMS). 
        `wfe_drift` is ignored.
    """
    
    # Get Pupil OPD info and convert to OTE LM
    opd_dict = self.get_opd_info(opd)
    opd_name = opd_dict['opd_name']
    opd_num  = opd_dict['opd_num']
    opd_str  = opd_dict['opd_str']
    opd      = deepcopy(opd_dict['pupilopd'])
        
    # Apply drift components
    wfe_dict = {'therm':0, 'frill':0, 'iec':0, 'opd':opd}
    if (wfe_therm is not None) or (wfe_frill is not None) or (wfe_iec is not None):
        wfe_therm = 0 if wfe_therm is None else wfe_therm
        wfe_frill = 0 if wfe_frill is None else wfe_frill
        wfe_iec = 0 if wfe_iec is None else wfe_iec

        # Apply IEC
        opd.apply_iec_drift(amplitude=wfe_iec, delay_update=True)
        # Apply frill
        opd.apply_frill_drift(amplitude=wfe_frill, delay_update=True)

        # Apply OTE thermal slew amplitude
        # This is slightly different due to how thermal slews are specified
        delta_time = 14*24*60 * u.min
        wfe_scale = (wfe_therm / 24)
        if wfe_scale == 0:
            delta_time = 0
        opd.thermal_slew(delta_time, case='BOL', scaling=wfe_scale)
        
        wfe_dict['therm'] = wfe_therm
        wfe_dict['frill'] = wfe_frill
        wfe_dict['iec']   = wfe_iec
        wfe_dict['opd']   = opd
    elif (wfe_drift != 0):
        _log.info('Performing WFE drift of {}nm'.format(wfe_drift))

        # Apply WFE drift to OTE Linear Model (Amplitude of frill drift)
        # self.pupilopd = opd
        # self.pupil = opd

        # Split WFE drift amplitude between three processes
        # 1) IEC Heaters; 2) Frill tensioning; 3) OTE Thermal perturbations
        # Give IEC heaters 1 nm 
        wfe_iec = 1 if np.abs(wfe_drift) > 2 else 0

        # Split remainder between frill and OTE thermal slew
        wfe_remain_var = wfe_drift**2 - wfe_iec**2
        wfe_frill = np.sqrt(0.8*wfe_remain_var)
        wfe_therm = np.sqrt(0.2*wfe_remain_var)
        # wfe_th_frill = np.sqrt((wfe_drift**2 - wfe_iec**2) / 2)

        # Negate amplitude if supplying negative wfe_drift
        if wfe_drift < 0:
            wfe_frill *= -1
            wfe_therm *= -1
            wfe_iec *= -1

        # Apply IEC
        opd.apply_iec_drift(amplitude=wfe_iec, delay_update=True)
        # Apply frill
        opd.apply_frill_drift(amplitude=wfe_frill, delay_update=True)

        # Apply OTE thermal slew amplitude
        # This is slightly different due to how thermal slews are specified
        delta_time = 14*24*60 * u.min
        wfe_scale = (wfe_therm / 24)
        if wfe_scale == 0:
            delta_time = 0
        opd.thermal_slew(delta_time, case='BOL', scaling=wfe_scale)
        
        wfe_dict['therm'] = wfe_therm
        wfe_dict['frill'] = wfe_frill
        wfe_dict['iec']   = wfe_iec
        wfe_dict['opd']   = opd
    else: # No drift
        # Apply IEC
        opd.apply_iec_drift(amplitude=0, delay_update=True)
        # Apply frill
        opd.apply_frill_drift(amplitude=0, delay_update=True)
        # Apply OTE thermal slew amplitude
        opd.thermal_slew(0*u.min, scaling=0)
        wfe_dict['opd'] = opd

    return wfe_dict

def _update_mask_shifts(self):
    """ 
    Restrict mask offsets to whole pixel shifts. Update source_offset_r/theta
    to accommodate sub-pixel shifts. Save sub-pixel shift values to temporary 
    source_offset_xsub/ysub keys in the options dict.
    """

    # Restrict mask offsets to whole pixel shifts
    # Subpixel offsets should be handled with source offsets
    xv = self.options.get('coron_shift_x', 0)
    yv = self.options.get('coron_shift_y', 0)

    if (not self.is_coron) or (xv==yv==0):
        return
        
    # Whole pixel offsets
    pixscale = self.pixelscale
    xv_pix = int(xv / pixscale) * pixscale
    yv_pix = int(yv / pixscale) * pixscale
    self.options['coron_shift_x'] = xv_pix # arcsec
    self.options['coron_shift_y'] = yv_pix # arcsec

    # Subpixel residuals
    xv_subpix = xv - xv_pix
    yv_subpix = yv - yv_pix

    rotation = 0 if self._rotation is None else -1*self._rotation
    # Equivalent source offsetting
    xoff_sub, yoff_sub = xy_rot(-1*xv_subpix, -1*yv_subpix, rotation)
    # Get initial values if they exist
    r0 = self.options.get('source_offset_r', 0)
    th0 = self.options.get('source_offset_theta', 0)
    x0, y0 = rtheta_to_xy(r0, th0)

    # Update (r, th) offsets
    r, th = xy_to_rtheta(x0+xoff_sub, y0+yoff_sub)
    self.options['source_offset_r'] = r
    self.options['source_offset_theta'] = th

    # print(xv, yv)
    # print('coron_shift_x: ', xv_pix, 'coron_shift_y: ', yv_pix)
    # print(xv_subpix, yv_subpix, rotation)
    # print(xoff_sub, yoff_sub)
    # print(r0, th0, x0, y0)
    # print('source_offset_r: ', r, 'source_offset_theta: ', th)

    # Store the xsub and ysub to back out later
    self.options['source_offset_xsub'] = xoff_sub # arcsec
    self.options['source_offset_ysub'] = yoff_sub # arcsec


def _calc_psf_with_shifts(self, calc_psf_func, **kwargs):
    """
    Mask shifting in webbpsf does not have as high of a precision as source offsetting
    for a given oversample setting.

    When performing mask shifting, coron_shift_x/y should be detector pixel integers. 
    Sub-pixel shifting should occur using source_offset_r/theta, but we must shift 
    the final images in the opposite direction in order to reposition the PSF at the
    proper source location. If the user already set _r/theta, the PSF is shifted to 
    that position using temporary source_offset_xsub/ysub keys in the options dict.
    """

    from .utils import S

    # Only shift coronagraphic masks by pixel integers
    # sub-pixels shifts are handled by source offsetting
    if self.is_coron:
        options_orig = self.options.copy()
        _update_mask_shifts(self)

    # Get spectrum; flat in photlam if not specified
    sp = kwargs.pop('sp', None)
    if sp is None:
        sp = stellar_spectrum('flat')
        do_counts = False
    else:
        do_counts = True
        
    # Create source weights
    bp = self.bandpass
    obs = S.Observation(sp, bp, binset=bp.wave)
    obs.convert('photlam')
    # Decide on wavelengths and weights
    npsf = self.npsf
    wave_um = np.linspace(obs.binwave.min(), obs.binwave.max(), npsf) / 1e4
    weights = np.interp(wave_um, obs.binwave/1e4, obs.binflux)
    weights /= weights.sum()
    src = {'wavelengths': wave_um*1e-6, 'weights': weights}

    # NIRCam grism pupils aren't recognized by WebbPSF
    if (self.name.upper()=='NIRCAM') and self.is_grism:
        grism_temp = self.pupil_mask
        self.pupil_mask = None

    # Perform PSF calculation
    kw_list = [
        'nlambda', 'monochromatic',
        'fov_arcsec', 'fov_pixels', 'oversample',
        'detector_oversample', 'fft_oversample',
        'outfile', 'overwrite', 'display',
        'save_intermediates', 'return_intermediates', 
        'normalize', 'add_distortion', 'crop_psf'
        ]
    kwargs2 = {}
    for kw in kw_list:
        if kw in kwargs.keys(): kwargs2[kw] = kwargs[kw]
    hdul = calc_psf_func(source=src, **kwargs2)

    # Return grism
    if (self.name.upper()=='NIRCAM') and self.is_grism:
        self.pupil_mask = grism_temp

    # Specify image oversampling relative to detector sampling
    for hdu in hdul:
        if 'DET' in hdu.header['EXTNAME']:
            osamp = 1
        else:
            osamp = hdu.header['DET_SAMP']
        hdu.header['OSAMP'] = (osamp, 'Image oversample vs det')

    # Scale PSF by total incident source flux
    if do_counts:
        for hdu in hdul:
            hdu.data *= obs.countrate()

    # Perform sub-pixel shifting to reposition PSF at requested source location
    if self.is_coron:
        for hdu in hdul:
            pix_scale = hdu.header['PIXELSCL']
            xoff_pix = self.options.get('source_offset_xsub',0) / pix_scale
            yoff_pix = self.options.get('source_offset_ysub',0) / pix_scale
            # hdu.data = fshift(hdu.data, -1*xoff_pix, -1*yoff_pix)
            hdu.data = fourier_imshift(hdu.data, -1*xoff_pix, -1*yoff_pix)

        # Return to previous values
        self.options = options_orig

    return hdul

def _calc_psf_webbpsf(self, calc_psf_func, add_distortion=None, fov_pixels=None, oversample=None, 
    wfe_drift=None, coord_vals=None, coord_frame='tel', **kwargs):

    """ Compute a WebbPSF PSF

    Slight modification of inherent WebbPSF `calc_psf` function. If add_distortion, fov_pixels,
    and oversample are not specified, then we automatically use the associated attributes.
    """

    # Automatically use already defined properties
    add_distortion = self.include_distortions if add_distortion is None else add_distortion
    fov_pixels = self.fov_pix if fov_pixels is None else fov_pixels

    kwargs['add_distortion'] = add_distortion
    kwargs['fov_pixels'] = fov_pixels

    # If there are distortion, compute some fov_pixels, which will get cropped later
    npix_extra = 5
    if add_distortion:
        kwargs['fov_pixels'] += 2 * npix_extra

    # Figure out sampling (always want >=4 for Lyot/coronagraphy)
    try:  # is_lyot may not always be a valid attribute
        is_lyot = self.is_lyot
    except AttributeError:
        is_lyot = False
    try:  # is_coron may not always be a valid attribute
        is_coron = self.is_coron
    except AttributeError:
        is_coron = False
    if is_lyot or is_coron:
        if oversample is None:
            if self.oversample>=4: # we're good!
                oversample = self.oversample
            else: # different oversample and detector_oversample
                _log.info(f'For coronagraphy, setting oversample=4 and detector_oversample={oversample}')
                kwargs['detector_oversample'] = self.oversample
                oversample = 4
        elif oversample<4: # no changes, but send informational message
            _log.warn('oversample={oversample} may produce imprecise results for coronagraphy. Suggest >=4.')
    else:
        oversample = self.oversample if oversample is None else oversample

    # Note: output image oversampling is overriden by 'detector_oversample'
    # This simply specifies FFT ovesampling
    kwargs['oversample'] = oversample

    # Drift OPD
    wfe_drift = 0 if wfe_drift is None else wfe_drift
    if wfe_drift != 0:
        # Create copies
        pupilopd_orig = deepcopy(self.pupilopd)
        pupil_orig    = deepcopy(self.pupil)

        # Get OPD info and convert to OTE LM
        opd_dict = self.get_opd_info(HDUL_to_OTELM=True)
        opd      = opd_dict['pupilopd']
        # Perform OPD drift and store in pupilopd and pupil attributes
        wfe_dict = self.drift_opd(wfe_drift, opd=opd)
        self.pupilopd = wfe_dict['opd']
        self.pupil    = wfe_dict['opd']

    # Get new sci coord
    if coord_vals is not None:
        siaf_ap = self.siaf[self.aperturename]
        xorig, yorig = self.detector_position
        xnew, ynew = coord_vals

        coron_shift_x_orig = self.options.get('coron_shift_x', 0)
        coron_shift_y_orig = self.options.get('coron_shift_y', 0)
        bar_offset_orig = self.options.get('bar_offset', None)
        self.options['bar_offset'] = 0

        # bar_offset_orig = self.options.get('b')

        xidl, yidl = siaf_ap.convert(xnew, ynew, coord_frame, 'idl')
        xsci, ysci = siaf_ap.convert(xnew, ynew, coord_frame, 'sci')
        self.detector_position = (xsci, ysci)

        # For coronagraphy, perform mask shift
        if self.is_coron:
            # Mask shift information

            field_rot = 0 if self._rotation is None else self._rotation

            # Convert to mask shifts (arcsec)
            # xoff_asec = (xsci - siaf_ap.XSciRef) * siaf_ap.XSciScale  # asec offset from reference
            # yoff_asec = (ysci - siaf_ap.YSciRef) * siaf_ap.YSciScale  # asec offset from reference
            xoff_asec, yoff_asec = (xidl, yidl)
            xoff_mask, yoff_mask = xy_rot(-1*xoff_asec, -1*yoff_asec, field_rot)

            # print(xnew, ynew, coord_frame)
            # print(xsci, ysci, siaf_ap.AperName)
            # print(siaf_ap.XSciRef, siaf_ap.YSciRef)
            # print(xoff_asec, yoff_asec)
            # print(xoff_mask, yoff_mask)

            # Shift mask in opposite direction
            self.options['coron_shift_x'] = xoff_mask
            self.options['coron_shift_y'] = yoff_mask

    # Perform PSF calculation
    return_hdul = kwargs.pop('return_hdul', True)
    return_oversample = kwargs.pop('return_oversample', True)
    hdul = _calc_psf_with_shifts(self, calc_psf_func, **kwargs)

    # Reset pupil and OPD
    if wfe_drift != 0:
        self.pupilopd = pupilopd_orig
        self.pupil    = pupil_orig

    # Return options
    if coord_vals is not None:
        self.detector_position = (xorig, yorig)
        self.options['coron_shift_x'] = coron_shift_x_orig
        self.options['coron_shift_y'] = coron_shift_y_orig
        self.options['bar_offset'] = bar_offset_orig

    # Crop distorted borders
    if add_distortion:
        osamp = hdul[0].header['DET_SAMP']
        npix_over = npix_extra * osamp
        hdul[0].data = hdul[0].data[npix_over:-npix_over,npix_over:-npix_over]
        hdul[2].data = hdul[2].data[npix_over:-npix_over,npix_over:-npix_over]
        hdul[1].data = hdul[1].data[npix_extra:-npix_extra,npix_extra:-npix_extra]
        hdul[3].data = hdul[3].data[npix_extra:-npix_extra,npix_extra:-npix_extra]

    # Check if we set return_hdul=False
    if return_hdul:
        res = hdul
    else:
        # If just returning a single image, determine oversample and distortion
        res = hdul[2].data if add_distortion else hdul[0].data
        if not return_oversample:
            res = frebin(res, scale=1/self.oversample)

    return res


def _inst_copy(self):
    """ Return a copy of the current instrument class. """

    # Change log levels to WARNING for webbpsf_ext, WebbPSF, and POPPY
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    init_params = {
        'filter'    : self.filter, 
        'pupil_mask': self.pupil_mask, 
        'image_mask': self.image_mask, 
        'fov_pix'   : self.fov_pix, 
        'oversample': self.oversample,
        'auto_gen_coeffs': False
    }

    # Init same subclass
    if self.name=='NIRCam':
        inst = NIRCam_ext(**init_params)
    elif self.name=='MIRI':
        inst = MIRI_ext(**init_params)

    # Get OPD info
    inst.pupilopd = deepcopy(self.pupilopd)
    inst.pupil    = deepcopy(self.pupil)

    # Detector and aperture info
    inst._detector = self._detector
    inst._detector_position = self._detector_position
    inst._aperturename = self._aperturename

    # Other options
    inst.options = self.options

    # PSF coeff info
    inst.use_legendre = self.use_legendre
    inst._ndeg = self._ndeg
    inst._npsf = self._npsf
    inst._quick = self._quick

    # SI WFE and distortions
    inst.include_si_wfe = self.include_si_wfe
    inst.include_distortions = self.include_distortions

    ### Instrument-specific parameters
    # Grism order for NIRCam
    try: inst._grism_order = self._grism_order
    except: pass

    # ND square for NIRCam
    try: inst._ND_acq = self._ND_acq
    except: pass

    setup_logging(log_prev, verbose=False)

    return inst


def _wrap_coeff_for_mp(args):
    """
    Internal helper routine for parallelizing computations across multiple processors
    for multiple WebbPSF monochromatic calculations.

    args => (inst,w,fov_pix,oversample)
    """
    # No multiprocessing for monochromatic wavelengths
    mp_prev = poppy.conf.use_multiprocessing
    poppy.conf.use_multiprocessing = False

    inst, w = args

    try:
        hdu_list = inst.calc_psf(monochromatic=w*1e-6, crop_psf=True)
    except Exception as e:
        print('Caught exception in worker thread (w = {}):'.format(w))
        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()

        print('')
        #raise e
        poppy.conf.use_multiprocessing = mp_prev
        return None

    # Return to previous setting
    poppy.conf.use_multiprocessing = mp_prev

    # Return distorted PSF
    if inst.include_distortions:
        hdu = hdu_list[2]
    else:
        hdu = hdu_list[0]

    # Rather than storing 
    hdu.header['OSAMP'] = (inst.oversample, 'Image oversample vs det')
    return hdu

def _gen_psf_coeff(self, nproc=None, wfe_drift=0, force=False, save=True, 
                   return_results=False, return_extras=False, **kwargs):

    """Generate PSF coefficients

    Creates a set of coefficients that will generate simulated PSFs for any
    arbitrary wavelength. This function first simulates a number of evenly-
    spaced PSFs throughout the specified bandpass (or the full channel). 
    An nth-degree polynomial is then fit to each oversampled pixel using 
    a linear-least squares fitting routine. The final set of coefficients 
    for each pixel is returned as an image cube. The returned set of 
    coefficient are then used to produce PSF via `calc_psf_from_coeff`.

    Useful for quickly generated imaging and dispersed PSFs for multiple
    spectral types. 

    Parameters
    ----------
    nproc : bool or None
        Manual setting of number of processor cores to break up PSF calculation.
        If set to None, this is determined based on the requested PSF size,
        number of available memory, and hardware processor cores. The automatic
        calculation endeavors to leave a number of resources available to the
        user so as to not crash the user's machine. 
    wfe_drift : float
        Wavefront error drift amplitude in nm.
    force : bool
        Forces a recalculation of PSF even if saved PSF exists. (default: False)
    save : bool
        Save the resulting PSF coefficients to a file? (default: True)
    return_results : bool
        By default, results are saved as object the attributes `psf_coeff` and
        `psf_coeff_header`. If return_results=True, results are instead returned
        as function outputs and will not be saved to the attributes. This is mostly
        used for successive coeff simulations to determine varying WFE drift or 
        focal plane dependencies.
    return_extras : bool
        Additionally returns a dictionary of monochromatic PSFs images and their 
        corresponding wavelengths for debugging purposes. Can be used with or without
        `return_results`. If `return_results=False`, then only this dictionary is
        returned, otherwise if `return_results=False` then returns everything as a
        3-element tuple (psf_coeff, psf_coeff_header, extras_dict).
    """

    save_name = self.save_name
    outfile = str(self.save_dir / save_name)
    # Load data from already saved FITS file
    if os.path.exists(outfile) and (not force):
        if return_extras:
            _log.warn("return_extras only valid if coefficient files does not exist or force=True")

        _log.info(f'Loading {outfile}')
        hdul = fits.open(outfile)
        data = hdul[0].data.astype(np.float)
        hdr  = hdul[0].header
        hdul.close()

        # Output if return_results=True, otherwise save to attributes
        if return_results:
            return data, hdr
        else:
            del self.psf_coeff, self.psf_coeff_header
            self.psf_coeff = data
            self.psf_coeff_header = hdr
            return
    
    temp_str = 'and saving' if save else 'but not saving'
    _log.info(f'Generating {temp_str} PSF coefficient')

    # Change log levels to WARNING for webbpsf_ext, WebbPSF, and POPPY
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    
    # w1 = self.bandpass.wave.min() / 1e4
    # w2 = self.bandpass.wave.max() / 1e4
    w1, w2 = self.wave_fit
    npsf = self.npsf
    waves = np.linspace(w1, w2, npsf)
        
    fov_pix = self.fov_pix 
    oversample = self.oversample 
            
    # Get OPD info and convert to OTE LM
    opd_dict = self.get_opd_info(HDUL_to_OTELM=True)
    opd_name = opd_dict['opd_name']
    opd_num  = opd_dict['opd_num']
    opd_str  = opd_dict['opd_str']
    opd      = opd_dict['pupilopd']
    
    # Drift OPD
    if wfe_drift!=0:
        wfe_dict = self.drift_opd(wfe_drift, opd=opd)
    else:
        wfe_dict = {'therm':0, 'frill':0, 'iec':0, 'opd':opd}
    opd_new = wfe_dict['opd']
    # Save copies
    pupilopd_orig = deepcopy(self.pupilopd)
    pupil_orig = deepcopy(self.pupil)
    self.pupilopd = opd_new
    self.pupil    = opd_new
    
    # How many processors to split into?
    if nproc is None:
        nproc = nproc_use(fov_pix, oversample, npsf)
    _log.debug('nprocessors: {}; npsf: {}'.format(nproc, npsf))

    # Make a paired down copy of self with limited data for 
    # copying to multiprocessor theads. This reduces memory
    # swapping overheads and limitations.
    inst_copy = _inst_copy(self) if nproc > 1 else self

    t0 = time.time()
    # Setup the multiprocessing pool and arguments to pass to each pool
    worker_arguments = [(inst_copy, wlen) for wlen in waves]
    if nproc > 1:

        hdu_arr = []
        try:
            with mp.Pool(nproc) as pool:
                for res in tqdm(pool.imap(_wrap_coeff_for_mp, worker_arguments), 
                                total=npsf, desc='Monochromatic PSFs', leave=False):
                    hdu_arr.append(res)
                pool.close()
            if hdu_arr[0] is None:
                raise RuntimeError('Returned None values. Issue with multiprocess or WebbPSF??')
        except Exception as e:
            _log.error('Caught an exception during multiprocess.')
            _log.info('Closing multiprocess pool.')
            pool.terminate()
            pool.close()
            raise e
        else:
            _log.info('Closing multiprocess pool.')
    else:
        # Pass arguments to the helper function
        hdu_arr = []
        for wa in tqdm(worker_arguments, desc='Monochromatic PSFs', leave=False):
            hdu = _wrap_coeff_for_mp(wa)
            if hdu is None:
                raise RuntimeError('Returned None values. Issue with WebbPSF??')
            hdu_arr.append(hdu)

    del inst_copy, worker_arguments
    t1 = time.time()

    # Ensure PSF sum is not larger than 1.0
    # This can sometimes occur for distorted PSFs near edges
    for hdu in hdu_arr:
        data_sum = hdu.data.sum()
        # print(data_sum)
        if data_sum>1:
            hdu.data /= data_sum
    
    # Reset pupils
    self.pupilopd = pupilopd_orig
    self.pupil = pupil_orig

    # Reset to original log levels
    setup_logging(log_prev, verbose=False)
    time_string = 'Took {:.2f} seconds to generate WebbPSF images'.format(t1-t0)
    _log.info(time_string)

    # Extract image data from HDU array
    images = []
    for hdu in hdu_arr:
        images.append(hdu.data)

    # Turn results into a numpy array (npsf,ny,nx)
    images = np.asarray(images)

    # Simultaneous polynomial fits to all pixels using linear least squares
    use_legendre = self.use_legendre
    ndeg = self.ndeg
    coeff_all = jl_poly_fit(waves, images, deg=ndeg, use_legendre=use_legendre, lxmap=[w1,w2])

    ################################
    # Create HDU and header
    ################################
    
    hdu = fits.PrimaryHDU(coeff_all)
    hdr = hdu.header
    head_temp = hdu_arr[0].header

    hdr['DESCR']    = ('PSF Coeffecients', 'File Description')
    hdr['NWAVES']   = (npsf, 'Number of wavelengths used in calculation')

    copy_keys = [
        'EXTNAME', 'OSAMP', 'OVERSAMP', 'DET_SAMP', 'PIXELSCL', 'FOV',     
        'INSTRUME', 'FILTER', 'PUPIL', 'CORONMSK',
        'WAVELEN', 'DIFFLMT', 'APERNAME', 'MODULE', 'CHANNEL', 'PILIN',
        'DET_NAME', 'DET_X', 'DET_Y', 'DET_V2', 'DET_V3',  
        'GRATNG14', 'GRATNG23', 'FLATTYPE', 'CCCSTATE', 'TACQNAME',
        'PUPILINT', 'PUPILOPD', 'OPD_FILE', 'OPDSLICE', 'TEL_WFE', 
        'SI_WFE', 'SIWFETYP', 'SIWFEFPT',
        'ROTATION', 'DISTORT', 'SIAF_VER', 'MIR_DIST', 'KERN_AMP', 'KERNFOLD',
        'NORMALIZ', 'FFTTYPE', 'AUTHOR', 'DATE', 'VERSION',  'DATAVERS'
    ]
    for key in copy_keys:
        try:
            hdr[key] = (head_temp[key], head_temp.comments[key])
        except (AttributeError, KeyError):
            pass
            # hdr[key] = ('none', 'No key found')
    hdr['WEXTVERS'] = (__version__, "webbpsf_ext version")
    # Update keywords
    hdr['PUPILOPD'] = (opd_name, 'Original Pupil OPD source')
    hdr['OPDSLICE'] = (opd_num, 'OPD slice index')

    # Source positioning
    offset_r = self.options.get('source_offset_r', 'None')
    offset_theta = self.options.get('source_offset_theta', 'None')
    
    # Mask offsetting
    coron_shift_x = self.options.get('coron_shift_x', 'None')
    coron_shift_y = self.options.get('coron_shift_y', 'None')
    bar_offset = self.options.get('bar_offset', 'None')
        
    # Jitter settings
    jitter = self.options.get('jitter')
    jitter_sigma = self.options.get('jitter_sigma', 0)
            
    # gen_psf_coeff() Keyword Values
    hdr['FOVPIX'] = (fov_pix, 'WebbPSF pixel FoV')
    hdr['NPSF']   = (npsf, 'Number of wavelengths to calc')
    hdr['NDEG']   = (ndeg, 'Polynomial fit degree')
    hdr['WAVE1']  = (w1, 'First wavelength in calc')
    hdr['WAVE2']  = (w2, 'Last of wavelength in calc')
    hdr['LEGNDR'] = (use_legendre, 'Legendre polynomial fit?')
    hdr['OFFR']  = (offset_r, 'Radial offset')
    hdr['OFFTH'] = (offset_theta, 'Position angle OFFR (CCW)')
    if (self.image_mask is not None) and ('WB' in self.image_mask):
        hdr['BAROFF'] = (bar_offset, 'Image mask shift along wedge (arcsec)')
    hdr['MASKOFFX'] = (coron_shift_x, 'Image mask shift in x (arcsec)')
    hdr['MASKOFFY'] = (coron_shift_y, 'Image mask shift in y (arcsec)')
    if jitter is None:
        hdr['JITRTYPE'] = ('None', 'Type of jitter applied')
    else:
        hdr['JITRTYPE'] = (jitter, 'Type of jitter applied')
    hdr['JITRSIGM'] = (jitter_sigma, 'Jitter sigma [mas]')
    if opd is None:
        hdr['OPD'] = ('None', 'Telescope OPD')
    elif isinstance(opd, fits.HDUList):
        hdr['OPD'] = ('HDUList', 'Telescope OPD')
    elif isinstance(opd, six.string_types):
        hdr['OPD'] = (opd, 'Telescope OPD')
    elif isinstance(opd, poppy.OpticalElement):
        hdr['OPD'] = ('OTE Linear Model', 'Telescope OPD')
    else:
        hdr['OPD'] = ('UNKNOWN', 'Telescope OPD')
    hdr['WFEDRIFT'] = (wfe_drift, "WFE drift amount [nm]")
    hdr['OTETHMDL'] = (opd._thermal_model.case, "OTE Thermal slew model case")
    hdr['OTETHSTA'] = ("None", "OTE Starting pitch angle for thermal slew model")
    hdr['OTETHEND'] = ("None", "OTE Ending pitch angle for thermal slew model")
    hdr['OTETHRDT'] = ("None", "OTE Thermal slew model delta time after slew")
    hdr['OTETHRWF'] = (wfe_dict['therm'], "OTE WFE amplitude from 'thermal slew' term")
    hdr['OTEFRLWF'] = (wfe_dict['frill'], "OTE WFE amplitude from 'frill tension' term")
    hdr['OTEIECWF'] = (wfe_dict['iec'],   "OTE WFE amplitude from 'IEC thermal cycling'")
    hdr['SIWFE']    = (self.include_si_wfe, "Was SI WFE included?")
    hdr['FORCE']    = (force, "Forced calculations?")
    hdr['SAVE']     = (save, "Was file saved to disk?")
    hdr['FILENAME'] = (save_name, "File save name")

    hdr.insert('WEXTVERS', '', after=True)
    hdr.insert('WEXTVERS', ('','gen_psf_coeff() Parameters'), after=True)
    hdr.insert('WEXTVERS', '', after=True)

    hdr.add_history(time_string)

    if save:
        # Catch warnings in case header comments too long
        from astropy.utils.exceptions import AstropyWarning
        import warnings
        _log.info(f'Saving to {outfile}')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            hdu.writeto(outfile, overwrite=True)

    if return_results==False:
        del self.psf_coeff, self.psf_coeff_header
        self.psf_coeff = coeff_all
        self.psf_coeff_header = hdr
    extras_dict = {'images' : images, 'waves': waves}

    # Options to return results from function
    if return_results:
        if return_extras:
            return coeff_all, hdr, extras_dict
        else:
            return coeff_all, hdr
    elif return_extras:
        return extras_dict
    else:
        return

def _gen_wfedrift_coeff(self, force=False, save=True, wfe_list=[0,1,2,5,10,20,40], 
                        return_results=False, return_raw=False, **kwargs):
    """ Fit WFE drift coefficients

    This function finds a relationship between PSF coefficients in the 
    presence of WFE drift. For a series of WFE drift values, we generate 
    corresponding PSF coefficients and fit a  polynomial relationship to 
    the residual values. This allows us to quickly modify a nominal set of 
    PSF image coefficients to generate a new PSF where the WFE has drifted 
    by some amplitude.
    
    It's Legendre's all the way down...

    Parameters
    ----------
    force : bool
        Forces a recalculation of coefficients even if saved file exists. 
        (default: False)
    save : bool
        Save the resulting PSF coefficients to a file? (default: True)

    Keyword Args
    ------------
    wfe_list : array-like
        A list of wavefront error drift values (nm) to calculate and fit.
        Default is [0,1,2,5,10,20,40], which covers the most-likely
        scenarios (1-5nm) while also covering a range of extreme drift
        values (10-40nm).
    return_results : bool
        By default, results are saved in `self._psf_coeff_mod` dictionary. 
        If return_results=True, results are instead returned as function outputs 
        and will not be saved to the dictionary attributes. 
    return_raw : bool
        Normally, we return the relation between PSF coefficients as a function
        of position. Instead this returns (as function outputs) the raw values
        prior to fitting. Final results will not be saved to the dictionary attributes.

    Example
    -------
    Generate PSF coefficient, WFE drift modifications, then
    create an undrifted and drifted PSF. (pseudo-code)

    >>> fpix, osamp = (128, 4)
    >>> coeff0 = gen_psf_coeff()
    >>> wfe_cf = gen_wfedrift_coeff()
    >>> psf0   = gen_image_from_coeff(coeff=coeff0)

    >>> # Drift the coefficients
    >>> wfe_drift = 5   # nm
    >>> cf_fit = wfe_cf.reshape([wfe_cf.shape[0], -1])
    >>> cf_mod = jl_poly(np.array([wfe_drift]), cf_fit).reshape(coeff0.shape)
    >>> coeff5nm = coeff + cf_mod
    >>> psf5nm = gen_image_from_coeff(coeff=coeff5nm)
    """
    # fov_pix should not be more than some size to preserve memory
    fov_max = self._fovmax_wfedrift if self.oversample<=4 else self._fovmax_wfedrift / 2 
    fov_pix_orig = self.fov_pix
    if self.fov_pix>fov_max:
        self.fov_pix = fov_max if (self.fov_pix % 2 == 0) else fov_max + 1

    # Name to save array of oversampled coefficients
    save_dir = self.save_dir
    save_name = os.path.splitext(self.save_name)[0] + '_wfedrift.npz'
    outname = str(save_dir / save_name)

    # Load file if it already exists
    if (not force) and os.path.exists(outname):
        # Return fov_pix to original size
        self.fov_pix = fov_pix_orig
        _log.info(f"Loading {outname}")
        out = np.load(outname)

        wfe_drift = out.get('wfe_drift')
        # Account for possibility that wfe_drift_off is None
        try:
            wfe_drift_off = out.get('wfe_drift_off')
        except ValueError:
            wfe_drift_off = None
        wfe_drift_lxmap = out.get('wfe_drift_lxmap')

        if return_results:
            return wfe_drift, wfe_drift_off, wfe_drift_lxmap
        else:
            self._psf_coeff_mod['wfe_drift'] = wfe_drift
            self._psf_coeff_mod['wfe_drift_off'] = wfe_drift_off
            self._psf_coeff_mod['wfe_drift_lxmap'] = wfe_drift_lxmap
            return

    _log.warn('Generating WFE Drift coefficients. This may take some time...')

    # Cycle through WFE drifts for fitting
    wfe_list = np.array(wfe_list)
    npos = len(wfe_list)

    # Warn if mask shifting is currently enabled (only when an image mask is present)
    for off_pos in ['coron_shift_x','coron_shift_y']:
        val = self.options.get(off_pos)
        if (self.image_mask is not None) and (val is not None) and (val != 0):
            _log.warn(f'{off_pos} is set to {val:.3f} arcsec. Should this be 0?')

    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    # Calculate residuals
    cf_wfe = []
    for wfe_drift in tqdm(wfe_list, leave=False, desc='WFE Drift'):
        cf, _ = self.gen_psf_coeff(wfe_drift=wfe_drift, force=True, save=False, return_results=True, **kwargs)
        cf_wfe.append(cf)
    cf_wfe = np.asarray(cf_wfe)

    # For coronagraphic observations, produce an off-axis PSF by turning off mask
    cf_fit_off = cf_wfe_off = None
    if self.is_coron:
        image_mask_orig = self.image_mask
        apername_orig = self._aperturename
        self.image_mask = None
        
        cf_wfe_off = []
        for wfe_drift in tqdm(wfe_list, leave=False, desc="Off-Axis"):
            cf, _ = self.gen_psf_coeff(wfe_drift=wfe_drift, force=True, save=False, return_results=True, **kwargs)
            cf_wfe_off.append(cf)
        cf_wfe_off = np.asarray(cf_wfe_off)
        # Return to original values
        self.image_mask = image_mask_orig
        self.aperturename = apername_orig

        # Return fov_pix to original size
        self.fov_pix = fov_pix_orig
        setup_logging(log_prev, verbose=False)
        if return_raw:
            return cf_wfe, cf_wfe_off, wfe_list

        # Get residuals
        cf_wfe_off = cf_wfe_off - cf_wfe_off[0]

        # Fit each pixel with a polynomial and save the coefficient
        cf_shape = cf_wfe_off.shape[1:]
        cf_wfe_off = cf_wfe_off.reshape([npos, -1])
        lxmap = np.array([np.min(wfe_list), np.max(wfe_list)])
        cf_fit_off = jl_poly_fit(wfe_list, cf_wfe_off, deg=4, use_legendre=True, lxmap=lxmap)
        cf_fit_off = cf_fit_off.reshape([-1, cf_shape[0], cf_shape[1], cf_shape[2]])

        del cf_wfe_off
        cf_wfe_off = None

    else:
        # Return fov_pix to original size
        self.fov_pix = fov_pix_orig
        setup_logging(log_prev, verbose=False)
        if return_raw:
            return cf_wfe, cf_wfe_off, wfe_list

    # Get residuals
    cf_wfe = cf_wfe - cf_wfe[0]

    # Fit each pixel with a polynomial and save the coefficient
    cf_shape = cf_wfe.shape[1:]
    cf_wfe = cf_wfe.reshape([npos, -1])
    lxmap = np.array([np.min(wfe_list), np.max(wfe_list)])
    cf_fit = jl_poly_fit(wfe_list, cf_wfe, deg=4, use_legendre=True, lxmap=lxmap)
    cf_fit = cf_fit.reshape([-1, cf_shape[0], cf_shape[1], cf_shape[2]])

    if save:
        _log.info(f"Saving to {outname}")
        np.savez(outname, wfe_drift=cf_fit, wfe_drift_off=cf_fit_off, wfe_drift_lxmap=lxmap)
    _log.info('Done.')

    # Options to return results from function
    if return_results:
        return cf_fit, cf_fit_off, lxmap
    else:
        self._psf_coeff_mod['wfe_drift'] = cf_fit
        self._psf_coeff_mod['wfe_drift_off'] = cf_fit_off
        self._psf_coeff_mod['wfe_drift_lxmap'] = lxmap


def _gen_wfefield_coeff(self, force=False, save=True, return_results=False, return_raw=False, **kwargs):
    """ Fit WFE field-dependent coefficients

    Find a relationship between field position and PSF coefficients for
    non-coronagraphic observations and when `include_si_wfe` is enabled.

    Parameters
    ----------
    force : bool
        Forces a recalculation of coefficients even if saved file exists. 
        (default: False)
    save : bool
        Save the resulting PSF coefficients to a file? (default: True)

    Keyword Args
    ------------
    return_results : bool
        By default, results are saved in `self._psf_coeff_mod` dictionary. 
        If return_results=True, results are instead returned as function outputs 
        and will not be saved to the dictionary attributes. 
    return_raw : bool
        Normally, we return the relation between PSF coefficients as a function
        of position. Instead this returns (as function outputs) the raw values
        prior to fitting. Final results will not be saved to the dictionary attributes.
    """

    if (self.include_si_wfe==False) or (self.is_coron):
        _log.info("Skipping WFE field dependence...")
        if self.include_si_wfe==False:
            _log.info("   `include_si_wfe` attribute is set to False.")
        if self.is_coron:
            _log.info(f"   {self.name} coronagraphic image mask is in place.")
        del self._psf_coeff_mod['si_field'] # Delete potentially large array
        self._psf_coeff_mod['si_field'] = None
        self._psf_coeff_mod['si_field_v2grid'] = None
        self._psf_coeff_mod['si_field_v3grid'] = None
        self._psf_coeff_mod['si_field_apname'] = None

        return

    # Delete potentially large array
    if (not return_raw) and (not return_results):
        del self._psf_coeff_mod['si_field'] 

    # fov_pix should not be more than some size to preserve memory
    fov_max = self._fovmax_wfefield if self.oversample<=4 else self._fovmax_wfefield / 2 
    fov_pix_orig = self.fov_pix
    if self.fov_pix>fov_max:
        self.fov_pix = fov_max if (self.fov_pix % 2 == 0) else fov_max + 1

    # Name to save array of oversampled coefficients
    save_dir = self.save_dir
    save_name = os.path.splitext(self.save_name)[0] + '_wfefields.npz'
    outname = str(save_dir / save_name)

    # Load file if it already exists
    if (not force) and os.path.exists(outname):
        # Return fov_pix to original size
        self.fov_pix = fov_pix_orig
        _log.info(f"Loading {outname}")
        out = np.load(outname)
        if return_results:
            return out['arr_0'], out['arr_1'], out['arr_2'], out['arr_3']
        else:
            self._psf_coeff_mod['si_field'] = out['arr_0']
            self._psf_coeff_mod['si_field_v2grid'] = out['arr_1']
            self._psf_coeff_mod['si_field_v3grid'] = out['arr_2']
            self._psf_coeff_mod['si_field_apname'] = out['arr_3'].flatten()[0]
            return

    _log.warn('Generating field-dependent coefficients. This may take some time...')

    # Cycle through a list of field points
    # These are the measured CV3 field positions
    zfile = 'si_zernikes_isim_cv3.fits'
    if self.name=='NIRCam':
        channel = 'LW' if 'long' in self.channel else 'SW'
        module = self.module

        # Check if NIRCam Lyot wedges are in place
        if self.is_lyot:
            if module=='B':
                raise NotImplementedError("There are no Full Frame SIAF apertures defined for Mod B coronagraphy")
            # These are extracted from Zemax models
            zfile = 'si_zernikes_coron_wfe.fits'

    # Read in measured SI Zernike data
    data_dir = self._WebbPSF_basepath
    zernike_file = os.path.join(data_dir, zfile)
    ztable_full = Table.read(zernike_file)

    if self.name=="NIRCam":
        inst_name = self.name + channel + module
    else:
        inst_name = self.name
    ind_inst = [inst_name in row['instrument'] for row in ztable_full] 
    ind_inst = np.where(ind_inst)

    # Grab measured V2 and V3 positions
    v2_all = np.array(ztable_full[ind_inst]['V2'].tolist())
    v3_all = np.array(ztable_full[ind_inst]['V3'].tolist())

    # Add detector corners
    # Want full detector footprint, not just subarray aperture
    if self.name=='NIRCam':
        pupil = self.pupil_mask
        v2_min, v2_max, v3_min, v3_max = NIRCam_V2V3_limits(module, channel=channel, pupil=pupil, rederive=True, border=1)
        igood = v3_all > v3_min
        v2_all = np.append(v2_all[igood], [v2_min, v2_max, v2_min, v2_max])
        v3_all = np.append(v3_all[igood], [v3_min, v3_min, v3_max, v3_max])
        npos = len(v2_all)
    else: # Other instrument detector fields are perhaps a little simpler
        # Specify the full frame apertures for grabbing corners of FoV
        if self.name=='MIRI':
            ap = self.siaf['MIRIM_FULL']
        else:
            raise NotImplementedError("Field Variations not implemented for {}".format(self.name))
        
        v2_ref, v3_ref = ap.corners('tel', False)
        # Add border margin of 1"
        v2_avg = np.mean(v2_ref)
        v2_ref[v2_ref<v2_avg] -= 1
        v2_ref[v2_ref>v2_avg] += 1
        v3_avg = np.mean(v3_ref)
        v3_ref[v3_ref<v3_avg] -= 1
        v3_ref[v3_ref>v3_avg] += 1
        # V2/V3 min and max convert to arcmin and add to v2_all/v3_all
        v2_min, v2_max = np.array([v2_ref.min(), v2_ref.max()]) / 60.
        v3_min, v3_max = np.array([v3_ref.min(), v3_ref.max()]) / 60.
        v2_all = np.append(v2_all, [v2_min, v2_max, v2_min, v2_max])
        v3_all = np.append(v3_all, [v3_min, v3_min, v3_max, v3_max])
        npos = len(v2_all)

    # Convert V2/V3 positions to sci coords for specified aperture
    apname = self.aperturename
    ap = self.siaf[apname]
    xsci_all, ysci_all = ap.convert(v2_all*60, v3_all*60, 'tel', 'sci')

    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    # Initial settings
    coeff0 = self.psf_coeff
    x0, y0 = self.detector_position

    # Calculate new coefficients at each position
    cf_fields = []
    # Create progress bar object
    pbar = tqdm(zip(xsci_all, ysci_all), total=npos, desc='Field Points')
    for xsci, ysci in pbar:
        # Update progress bar description
        pbar.set_description(f"xsci, ysci = ({xsci:.0f}, {ysci:.0f})")
        # Update saved detector position and calculate PSF coeff
        self.detector_position = (xsci, ysci)
        cf, _ = self.gen_psf_coeff(force=True, save=False, return_results=True, **kwargs)
        cf_fields.append(cf)
    cf_fields = np.asarray(cf_fields)

    # Reset to initial values
    self.detector_position = (x0,y0)
    self.fov_pix = fov_pix_orig
    setup_logging(log_prev, verbose=False)

    # Return raw results for further analysis
    if return_raw:
        return cf_fields, v2_all, v3_all

    # Get residuals
    new_shape = cf_fields.shape[-2:]
    if coeff0.shape[-2:] != new_shape:
        coeff0_resize = np.asarray([pad_or_cut_to_size(im, new_shape) for im in coeff0])
        coeff0 = coeff0_resize

    cf_fields_resid = cf_fields - coeff0
    del cf_fields

    # Create an evenly spaced grid of V2/V3 coordinates
    nv23 = 8
    v2grid = np.linspace(v2_min, v2_max, num=nv23)
    v3grid = np.linspace(v3_min, v3_max, num=nv23)

    # Interpolate onto an evenly space grid
    res = make_coeff_resid_grid(v2_all, v3_all, cf_fields_resid, v2grid, v3grid)
    if save: 
        _log.info(f"Saving to {outname}")
        np.savez(outname, res, v2grid, v3grid, apname)

    if return_results:
        return res, v2grid, v3grid, apname
    else:
        self._psf_coeff_mod['si_field'] = res
        self._psf_coeff_mod['si_field_v2grid'] = v2grid
        self._psf_coeff_mod['si_field_v3grid'] = v3grid
        self._psf_coeff_mod['si_field_apname'] = apname


def _gen_wfemask_coeff(self, force=False, save=True, large_grid=None,
                       return_results=False, return_raw=False, **kwargs):

    if (not self.is_coron):
        _log.info("Skipping WFE mask dependence...")
        _log.info("   Coronagraphic image mask not in place")
        del self._psf_coeff_mod['si_mask'] # Delete potentially large array
        self._psf_coeff_mod['si_mask'] = None
        self._psf_coeff_mod['si_mask_xgrid'] = None
        self._psf_coeff_mod['si_mask_ygrid'] = None
        self._psf_coeff_mod['si_mask_apname'] = None
        return

    # Delete potentially large array
    if (not return_raw) and (not return_results):
        del self._psf_coeff_mod['si_mask'] 

    large_grid = self._psf_coeff_mod['si_mask_large'] if large_grid is None else large_grid

    # fov_pix should not be more than some size to preserve memory
    fov_max = self._fovmax_wfemask if self.oversample<=4 else self._fovmax_wfemask / 2 
    fov_pix_orig = self.fov_pix
    if self.fov_pix>fov_max:
        self.fov_pix = fov_max if (self.fov_pix % 2 == 0) else fov_max + 1

    # Modify bar_offset to always be 0 (center of wedge FOV)
    # Do this here so that it is captured in the save name
    bar_offset_orig = self.options.get('bar_offset', None)
    if (self.name=='NIRCam'):
        self.options['bar_offset'] = 0

    # Name to save array of oversampled coefficients
    save_dir = self.save_dir
    file_ext = '_large_grid_wfemask.npz' if large_grid else '_wfemask.npz'
    save_name = os.path.splitext(self.save_name)[0] + file_ext
    outname = str(save_dir / save_name)

    # Load file if it already exists
    if (not force) and os.path.exists(outname):
        # Return parameter to original
        self.fov_pix = fov_pix_orig
        if self.name=='NIRCam':
            self.options['bar_offset'] = bar_offset_orig

        _log.info(f"Loading {outname}")
        out = np.load(outname)
        if return_results:
            return out['arr_0'], out['arr_1'], out['arr_2'], out['arr_3']
        else:
            self._psf_coeff_mod['si_mask'] = out['arr_0']
            self._psf_coeff_mod['si_mask_xgrid'] = out['arr_1']
            self._psf_coeff_mod['si_mask_ygrid'] = out['arr_2']
            self._psf_coeff_mod['si_mask_apname'] = out['arr_3'].flatten()[0]
            self._psf_coeff_mod['si_mask_large'] = large_grid
            return

    if large_grid:
        _log.warn('Generating mask position-dependent coeffs (large grid). This may take some time...')
    else:
        _log.warn('Generating mask position-dependent coeffs (small grid). This may take some time...')

    # Current mask positions to return to at end
    coron_shift_x_orig = self.options.get('coron_shift_x', 0)
    coron_shift_y_orig = self.options.get('coron_shift_y', 0)
    detector_position_orig = self.detector_position
    apname = self.aperturename

    # Cycle through a list of field points
    if self.name=='MIRI':
        # Series of x and y mask shifts (in mask coordinates)
        # Negative shifts will place source in upper right quadrant
        # Depend on PSF symmetries for other three quadrants
        if 'FQPM' in self.image_mask:
            if large_grid:
                xy_offsets = -1 * np.array([0, 0.005, 0.01, 0.08, 0.10, 0.2, 0.5, 1, 5, 11])
            else:
                xy_offsets = -1 * np.array([0, 0.01, 0.10, 1, 10])
            xy_offsets[0] = 0
        elif 'LYOT' in self.image_mask:
            # TODO: Update offsets to optimize for Lyot mask
            if large_grid:
                xy_offsets = -1 * np.array([0, 0.01, 0.1, 0.36, 0.5, 1, 2.1, 5, 11])
            else:
                xy_offsets = -1 * np.array([0, 0.01, 0.10, 1, 10])
            xy_offsets[0] = 0
        else:
            raise NotImplementedError(f'{self.name} with {self.image_mask} not implemented.')

        x_offsets = y_offsets = np.sort(xy_offsets) # Ascending order
        grid_vals = np.array(np.meshgrid(y_offsets,x_offsets))
        xy_list = [(x,y) for x,y in grid_vals.reshape([2,-1]).transpose()]
        xoff, yoff = np.array(xy_list).transpose()

        # Small grid dithers indices
        # Always calculate these explicitly
        ind_zero = (np.abs(xoff)==0) & (np.abs(yoff)==0)
        iwa = 0.01
        ind_sgd = (np.abs(xoff)<=iwa) & (np.abs(yoff)<=iwa) & ~ind_zero
    elif self.name=='NIRCam':
        # Turn off ND square calculations
        # Such PSFs will be attenuated later
        nd_squares_orig = self.options.get('nd_squares', True)
        self.options['nd_squares'] = False

        # Build position offsets
        if self.image_mask[-1]=='R': # Round masks
            if large_grid:
                # Include SGD points
                xy_inner = np.array([0.015, 0.02, 0.05, 0.1])
                # M430R sampling; scale others
                # xy_mid = np.array([0.6, 1.2, 2, 2.5])
                xy_mid = np.array([0.6, 1.2, 2.5])
                if '210R' in self.image_mask:
                    xy_mid *= 0.488
                elif '335R' in self.image_mask:
                    xy_mid *= 0.779
                # xy_outer = np.array([5.0, 8.0])
                xy_outer = np.array([8.0])

                # Sort offsets [-], 0, [+]
                xy_pos = np.concatenate((xy_inner, xy_mid, xy_outer))
                xy_neg = -1 * xy_pos[::-1]
                xy_offsets = np.concatenate((xy_neg, [0], xy_pos))
            else:
                # Assume symmetries for round mask
                xy_offsets = np.array([-8, -1.5, -0.1, -0.01, 0])

            # Create grid spacing
            x_offsets = y_offsets = np.sort(xy_offsets)
            grid_vals = np.array(np.meshgrid(x_offsets,y_offsets))
            xy_list = [(x,y) for x,y in grid_vals.reshape([2,-1]).transpose()]
            xoff, yoff = np.array(xy_list).transpose()

            # Small grid dithers indices and close IWA
            # Always calculate these explicitly
            # Exclude x,y=(0,0) since we want to calc this early
            ind_zero = (np.abs(xoff)==0) & (np.abs(yoff)==0)
            iwa = 0.1
            ind_sgd = (np.abs(xoff)<=iwa) & (np.abs(yoff)<=iwa) & ~ind_zero
        else: # Bar masks
            if large_grid:
                # Include SGD points
                y_inner = np.array([0.01, 0.02, 0.05, 0.1])
                # LWB sampling of wedge gradient
                y_mid = np.array([0.6, 1.2, 2, 2.5])
                if 'SW' in self.image_mask:
                    y_mid *= 0.488
                y_outer = np.array([5,8])
                y_offsets = np.concatenate([y_inner, y_mid, y_outer])
                x_offsets = np.array([-8, -6, -4, -2, 0, 2, 4, 6, 8], dtype='float')

                # Mask offset values in ascending order
                x_offsets = np.sort(-1*x_offsets)
                y_offsets = np.concatenate([-1*y_offsets,[0],y_offsets])
                y_offsets = np.sort(-1*y_offsets)
            else:
                y_offsets = np.array([-8, -1.5, -0.1, -0.01, 0])
                x_offsets = np.array([-8, -5, -2, 0, 2, 5, 8], dtype='float')
            grid_vals = np.array(np.meshgrid(x_offsets,y_offsets))
            xy_list = [(x,y) for x,y in grid_vals.reshape([2,-1]).transpose()]
            xoff, yoff = np.array(xy_list).transpose()

            # Small grid dithers indices and close IWA
            # Always calculate these explicitly
            ind_zero = (np.abs(xoff)==0) & (np.abs(yoff)==0)
            iwa = 0.1
            ind_sgd = (np.abs(yoff)<=iwa) & ~ind_zero

    # Get PSF coefficients for each specified position
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    npos = len(xoff)
    fov_pix_over = self.fov_pix * self.oversample
    cf_all = np.zeros([npos, self.ndeg+1, fov_pix_over, fov_pix_over], dtype='float')
    # Create progress bar object
    pbar = trange(npos, leave=False, desc="Mask Offsets")
    for i in pbar:
        xv, yv = (xoff[i], yoff[i])
        # Update descriptive label
        pbar.set_description(f"xoff, yoff = ({xv:.2f}, {yv:.2f})")

        self.options['coron_shift_x'] = xv
        self.options['coron_shift_y'] = yv

        # Pixel offset information
        field_rot = 0 if self._rotation is None else self._rotation
        xyoff_pix = np.array(xy_rot(-1*xv, -1*yv, -1*field_rot)) / self.pixelscale
        self.detector_position = np.array(detector_position_orig) + xyoff_pix

        # Skip SGD locations until later
        if ind_sgd[i]==False:
            cf, _ = self.gen_psf_coeff(return_results=True, force=True, save=False, **kwargs)
            cf_all[i] = cf
            # Save central coefficient to it's own variable
            if (xv==0) and (yv==0):
                coeff0 = cf

    setup_logging(log_prev, verbose=False)

    # Return raw results for further analysis
    # Excludes concatenation of symmetric PSFs and SGD calculations
    if return_raw:
        # Return to previous values
        self.options['coron_shift_x'] = coron_shift_x_orig
        self.options['coron_shift_y'] = coron_shift_y_orig
        self.detector_position = detector_position_orig
        self.fov_pix = fov_pix_orig
        if self.name=='NIRCam':
            self.options['nd_squares'] = nd_squares_orig
            self.options['bar_offset'] = bar_offset_orig

        return cf_all, xoff, yoff

    # Get residuals
    cf_all -= coeff0
    cf_resid = cf_all
    # cf_resid = cf_all - coeff0
    # del cf_all

    # Reshape into cf_resid into [nypos, nxpos, ncf, nypix, nxpix]
    nxpos = len(x_offsets)
    nypos = len(y_offsets)
    xoff = xoff.reshape([nypos,nxpos])
    yoff = yoff.reshape([nypos,nxpos])
    sh = cf_resid.shape
    cf_resid = cf_resid.reshape([nypos,nxpos,sh[1],sh[2],sh[3]])
    ncf, nypix, nxpix = cf_resid.shape[-3:]

    # MIRI quadrant symmetries 
    # This doesn't work for NIRCam, because of chromatic PSF shifts in y-direction
    if self.name=='MIRI':

        field_rot = 0 if self._rotation is None else 2*self._rotation

        # Assuming that x=y=0 are in the final index (i=-1)

        # Add same x, but -1*y
        x_negy  = xoff[:-1,:]
        y_negy  = -1*yoff[:-1,:][::-1,:]
        cf_negy = cf_resid[:-1,:][::-1,:]
        sh_ret  = cf_negy.shape
        # Flip the PSF coeff image in the y-axis and rotate
        cf_negy = cf_negy[:,:,:,::-1,:].reshape([-1,nypix,nxpix])
        cf_negy = rotate_offset(cf_negy, field_rot, reshape=False, order=2, mode='mirror')
        cf_negy = cf_negy.reshape(sh_ret)

        # Add same y, but -1*x
        x_negx  = -1*xoff[:,:-1][:,::-1]
        y_negx  = yoff[:,:-1]
        cf_negx = cf_resid[:,:-1][:,::-1]
        sh_ret  = cf_negx.shape
        # Flip the PSF coeff image in the x-axis and rotate
        cf_negx = cf_negx[:,:,:,:,::-1].reshape([-1,nypix,nxpix])
        cf_negx = rotate_offset(cf_negx, field_rot, reshape=False, order=2, mode='mirror')
        cf_negx = cf_negx.reshape(sh_ret)

        # Add -1*y, -1*x; exclude all x=0 and y=0 coords
        x_negxy  = -1*xoff[:-1,:-1][::-1,::-1]
        y_negxy  = -1*yoff[:-1,:-1][::-1,::-1]
        cf_negxy = cf_resid[:-1,:-1][::-1,::-1]
        # Flip the PSF coeff image in both x-axis and y-axis
        # No rotation necessary
        cf_negxy = cf_negxy[:,:,:,::-1,::-1]        

        # Combine quadrants
        xoff1 = np.concatenate((xoff, x_negy), axis=0)
        xoff2 = np.concatenate((x_negx, x_negxy), axis=0)
        xoff_all = np.concatenate((xoff1, xoff2), axis=1)

        yoff1 = np.concatenate((yoff, y_negy), axis=0)
        yoff2 = np.concatenate((y_negx, y_negxy), axis=0)
        yoff_all = np.concatenate((yoff1, yoff2), axis=1)

        # Get rid of unnecessary and potentially large arrays
        del xoff1, xoff2, yoff1, yoff2

        cf1 = np.concatenate((cf_resid, cf_negy), axis=0)
        del cf_resid, cf_negy
        cf2 = np.concatenate((cf_negx, cf_negxy), axis=0)
        del cf_negx, cf_negxy
        cf_resid_all = np.concatenate((cf1, cf2), axis=1)
        del cf1, cf2

        # Get all SGD positions now that we've combined all x/y positions
        # For SGD regions, we want to calculate actual PSFs, not take
        # the shortcuts that were done above
        ind_zero = (np.abs(xoff_all)==0) & (np.abs(yoff_all)==0)
        ind_sgd = (np.abs(xoff_all)<=iwa) & (np.abs(yoff_all)<=iwa) & ~ind_zero

    elif (self.name=='NIRCam') and (self.image_mask[-1]=='R') and (large_grid==True):
        # Round Masks
        xoff_all = xoff
        yoff_all = yoff
        cf_resid_all = cf_resid

        # SGD positions
        ind_zero = (np.abs(xoff_all)==0) & (np.abs(yoff_all)==0)
        ind_sgd = (np.abs(xoff_all)<=iwa) & (np.abs(yoff_all)<=iwa) & ~ind_zero

    elif (self.name=='NIRCam') and (self.image_mask[-1]=='B') and (large_grid==True): 
        # Bar Masks
        xoff_all = xoff
        yoff_all = yoff
        cf_resid_all = cf_resid

        # SGD positions
        ind_zero = (np.abs(xoff_all)==0) & (np.abs(yoff_all)==0)
        ind_sgd = (np.abs(yoff_all)<=iwa) & ~ind_zero

    # Short cuts for quicker creation
    elif (self.name=='NIRCam') and (self.image_mask[-1]=='R') and (large_grid==False):

        # No need to rotate NIRcam, because self._rotation is None
        field_rot = 0 if self._rotation is None else 2*self._rotation

        # Assuming that x=y=0 are in the final index (i=-1)

        # Add same x, but -1*y
        x_negy  = xoff[:-1,:]
        y_negy  = -1*yoff[:-1,:][::-1,:]
        cf_negy = cf_resid[:-1,:][::-1,:]
        # Don't Flip the PSF coeff image in the y-axis
        # No rotation necessary

        # Add same y, but -1*x
        x_negx  = -1*xoff[:,:-1][:,::-1]
        y_negx  = yoff[:,:-1]
        cf_negx = cf_resid[:,:-1][:,::-1]
        # Flip the PSF coeff image in the x-axis and rotate
        cf_negx = cf_negx[:,:,:,:,::-1]
        if np.abs(field_rot) < 0.01:
            sh_ret  = cf_negx.shape
            cf_negx = cf_negx.reshape([-1,nypix,nxpix])
            cf_negx = rotate_offset(cf_negx, field_rot, reshape=False, order=2, mode='mirror')
            cf_negx = cf_negx.reshape(sh_ret)

        # Add -1*y, -1*x; exclude all x=0 and y=0 coords
        x_negxy  = -1*xoff[:-1,:-1][::-1,::-1]
        y_negxy  = -1*yoff[:-1,:-1][::-1,::-1]
        cf_negxy = cf_resid[:-1,:-1][::-1,::-1]
        # Flip the PSF coeff image only along x-axis
        # Rotate if necessary
        cf_negxy = cf_negxy[:,:,:,:,::-1]
        if np.abs(field_rot) < 0.01:
            sh_ret   = cf_negxy.shape
            cf_negxy = cf_negxy.reshape([-1,nypix,nxpix])
            cf_negxy = rotate_offset(cf_negxy, field_rot, reshape=False, order=2, mode='mirror')
            cf_negxy = cf_negxy.reshape(sh_ret)

        # Combine quadrants
        xoff1 = np.concatenate((xoff, x_negy), axis=0)
        xoff2 = np.concatenate((x_negx, x_negxy), axis=0)
        xoff_all = np.concatenate((xoff1, xoff2), axis=1)

        yoff1 = np.concatenate((yoff, y_negy), axis=0)
        yoff2 = np.concatenate((y_negx, y_negxy), axis=0)
        yoff_all = np.concatenate((yoff1, yoff2), axis=1)

        # Get rid of unnecessary and potentially large arrays
        del xoff1, xoff2, yoff1, yoff2

        cf1 = np.concatenate((cf_resid, cf_negy), axis=0)
        del cf_resid, cf_negy
        cf2 = np.concatenate((cf_negx, cf_negxy), axis=0)
        del cf_negx, cf_negxy
        cf_resid_all = np.concatenate((cf1, cf2), axis=1)
        del cf1, cf2

        # Get all SGD positions now that we've combined all x/y positions
        # For SGD regions, we want to calculate actual PSFs, not take
        # the shortcuts that were done above
        ind_zero = (np.abs(xoff_all)==0) & (np.abs(yoff_all)==0)
        ind_sgd = (np.abs(xoff_all)<=iwa) & (np.abs(yoff_all)<=iwa) & ~ind_zero

    # Short cuts for quicker creation
    elif (self.name=='NIRCam') and (self.image_mask[-1]=='B') and (large_grid==False): 
        # Bar masks

        # No need to rotate NIRcam, because self._rotation is None
        field_rot = 0 if self._rotation is None else 2*self._rotation

        # Assuming that y=0 are in the final index (i=-1)

        # Add same x, but -1*y
        x_negy  = xoff[:-1,:]
        y_negy  = -1*yoff[:-1,:][::-1,:]
        cf_negy = cf_resid[:-1,:][::-1,:]
        # Flip the PSF coeff image in the y-axis
        cf_negy = cf_negy[:,:,:,::-1,:]

        # Rotation
        if np.abs(field_rot) < 0.01:
            sh_ret  = cf_negy.shape
            cf_negy = cf_negy.reshape([-1,nypix,nxpix])
            cf_negy = rotate_offset(cf_negy, field_rot, reshape=False, order=2, mode='mirror')
            cf_negy = cf_negy.reshape(sh_ret)

        # Combine halves
        xoff_all = np.concatenate((xoff, x_negy), axis=0)
        yoff_all = np.concatenate((yoff, y_negy), axis=0)
        cf_resid_all = np.concatenate((cf_resid, cf_negy), axis=0)
        # Get rid of unnecessary and potentially large arrays
        del xoff, x_negy, yoff, y_negy, cf_resid, cf_negy

        # Get all SGD positions now that we've combined all x/y positions
        # For SGD regions, we want to calculate actual PSFs, not take
        # the shortcuts that were done above
        ind_zero = (np.abs(xoff_all)==0) & (np.abs(yoff_all)==0)
        ind_sgd = (np.abs(yoff_all)<=iwa) & ~ind_zero
    else:
        msg = f'{self.name} not implemented for different WFE mask modifications.'
        raise NotImplementedError(msg)

    # Get PSF coefficients for each SGD position
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    # Set to fov calculation size
    xsgd, ysgd = (xoff_all[ind_sgd], yoff_all[ind_sgd])
    nsgd = len(xsgd)
    if nsgd>0:
        # Create progress bar object
        pbar = tqdm(zip(xsgd, ysgd), total=nsgd, desc='SGD', leave=False)
        for xv, yv in pbar:
            # Update descriptive label
            pbar.set_description(f"xsgd, ysgd = ({xv:.2f}, {yv:.2f})")

            self.options['coron_shift_x'] = xv
            self.options['coron_shift_y'] = yv

            # Pixel offset information
            field_rot = 0 if self._rotation is None else self._rotation
            xyoff_pix = np.array(xy_rot(-1*xv, -1*yv, -field_rot)) / self.pixelscale
            self.detector_position = np.array(detector_position_orig) + xyoff_pix

            cf, _ = self.gen_psf_coeff(return_results=True, force=True, save=False, **kwargs)
            ind = (xoff_all==xv) & (yoff_all==yv)
            cf_resid_all[ind] = cf - coeff0

    setup_logging(log_prev, verbose=False)

    # Return to previous values
    self.options['coron_shift_x'] = coron_shift_x_orig
    self.options['coron_shift_y'] = coron_shift_y_orig
    self.detector_position = detector_position_orig
    self.fov_pix = fov_pix_orig
    if self.name=='NIRCam':
        self.options['nd_squares'] = nd_squares_orig
        self.options['bar_offset'] = bar_offset_orig

    # x and y grid values to return
    xvals = xoff_all[0]
    yvals = yoff_all[:,0]

    # Use field_coeff_func with x and y shift values
    #   xvals_new = np.array([-1.2, 4.0,  0.1, -3])
    #   yvals_new = np.array([ 3.0, 2.3, -0.1,  0])
    #   test = field_coeff_func(xvals, yvals, cf_resid_all, xvals_new, yvals_new)
    
    if save: 
        _log.info(f"Saving to {outname}")
        np.savez(outname, cf_resid_all, xvals, yvals, apname)

    if return_results:
        return cf_resid_all, xvals, yvals
    else:
        self._psf_coeff_mod['si_mask'] = cf_resid_all
        self._psf_coeff_mod['si_mask_xgrid'] = xvals
        self._psf_coeff_mod['si_mask_ygrid'] = yvals
        self._psf_coeff_mod['si_mask_apname'] = apname
        self._psf_coeff_mod['si_mask_large'] = large_grid



def _calc_psf_from_coeff(self, sp=None, return_oversample=True, return_hdul=True,
    wfe_drift=None, coord_vals=None, coord_frame='tel', break_iter=True, 
    siaf_ap=None, **kwargs):
    """PSF Image from polynomial coefficients
    
    Create a PSF image from instrument settings. The image is noiseless and
    doesn't take into account any non-linearity or saturation effects, but is
    convolved with the instrument throughput. Pixel values are in counts/sec.
    The result is effectively an idealized slope image (no background).

    If no spectral dispersers (grisms or DHS), then this returns a single
    image or list of images if sp is a list of spectra. By default, it returns
    only the oversampled PSF, but setting return_oversample=False will
    instead return a set of detector-sampled images.

    Parameters
    ----------
    sp : :mod:`pysynphot.spectrum`
        If not specified, the default is flat in phot lam (equal number of photons 
        per wavelength bin). The default is normalized to produce 1 count/sec within 
        that bandpass, assuming the telescope collecting area and instrument bandpass. 
        Coronagraphic PSFs will further decrease this due to the smaller pupil
        size and suppression of coronagraphic mask. 
        If set, then the resulting PSF image will be scaled to generate the total
        observed number of photons from the spectrum.
    return_oversample : bool
        If True, then also returns the oversampled version of the PSF (default: True).
    wfe_drift : float or None
        Wavefront error drift amplitude in nm.
    coord_vals : tuple or None
        Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
        If multiple values, then this should be an array ([xvals], [yvals]).
        Relative to self.aperturename aperture.
    coord_frame : str
        Type of input coordinates relative to self.aperturename aperture.

            * 'tel': arcsecs V2,V3
            * 'sci': pixels, in conventional DMS axes orientation
            * 'det': pixels, in raw detector read out axes orientation
            * 'idl': arcsecs relative to aperture reference location.

    return_hdul : bool
        Return PSFs in an HDUList rather than set of arrays (default: True).
    break_iter : bool
        For multiple field points, break up and generate PSFs one-by-one rather 
        than simultaneously, which can save on memory for large PSFs.
    """        

    psf_coeff_hdr = self.psf_coeff_header
    psf_coeff     = self.psf_coeff

    if psf_coeff is None:
        _log.warning("You must first run `gen_psf_coeff` in order to calculate PSFs.")
        return

    # Spectrographic Mode?
    is_spec = False
    if (self.name=='NIRCam') or (self.name=='NIRISS'):
        is_spec = True if self.is_grism else False
    elif (self.name=='MIRI') or (self.name=='NIRSpec'):
        is_spec = True if self.is_slitspec else False

    # Make sp a list of spectral objects if it already isn't
    if sp is None:
        nspec = 0
    elif (sp is not None) and (not isinstance(sp, list)):
        sp = [sp]
        nspec = 1
    else:
        nspec = len(sp)

    if coord_vals is not None:
        coord_vals = np.array(coord_vals)

    # If large number of requested field points, then break into single event calls
    if coord_vals is not None:
        c1_all, c2_all = coord_vals
        nfield_init = np.size(c1_all)
        break_iter = False if nfield_init<=1 else break_iter
        if break_iter:
            kwargs['sp'] = sp
            kwargs['return_oversample'] = return_oversample
            kwargs['return_hdul'] = return_hdul
            kwargs['wfe_drift'] = wfe_drift
            kwargs['coord_frame'] = coord_frame
            kwargs['siaf_ap'] = siaf_ap
            psf_all = fits.HDUList() if return_hdul else []
            for ii in trange(nfield_init, leave=False, desc='PSFs'):
                kwargs['coord_vals'] = (c1_all[ii], c2_all[ii])

                # Just a single spectrum? Or unique spectrum at each field point?
                kwargs['sp'] = sp[ii] if((sp is not None) and (nspec==nfield_init)) else sp

                res = _calc_psf_from_coeff(self, **kwargs)
                if return_hdul:
                    # For grisms (etc), the wavelength solution is the same for each field point
                    psf = res[0]
                    wave = res[1] if is_spec else None
                    # if ii>0:
                    #     psf = fits.ImageHDU(data=psf.data, header=psf.header)
                else:
                    # For grisms (etc), the wavelength solution is the same for each field point
                    wave, psf = res if is_spec else (None, res)
                psf_all.append(psf)
                
            if return_hdul:
                output = fits.HDUList(psf_all)
                if is_spec:
                    output.append(wave)
            else:
                output = np.asarray(psf_all)
                output = (wave, psf_all) if is_spec else output

            return output


    # Coeff modification variable
    psf_coeff_mod = 0 

    if wfe_drift is None: 
        wfe_drift = 0

    # Modify PSF coefficients based on field-dependence
    # Ignore if there is a focal plane mask
    # No need for SI WFE field dependence if coronagraphy, but this allows us
    # to enable `include_si_wfe` for NIRCam PSF calculation
    nfield = None
    if self.image_mask is None:
        cf_mod, nfield = _coeff_mod_wfe_field(self, coord_vals, coord_frame, 
                                              siaf_ap=siaf_ap)
        psf_coeff_mod += cf_mod
    nfield = 1 if nfield is None else nfield

    # Modify PSF coefficients based on field-dependence with a focal plane mask
    nfield_mask = None
    if self.image_mask is not None:
        cf_mod, nfield_mask = _coeff_mod_wfe_mask(self, coord_vals, coord_frame,
                                                  siaf_ap=siaf_ap)
        psf_coeff_mod += cf_mod
    nfield = nfield if nfield_mask is None else nfield_mask

    # Modify PSF coefficients based on WFE drift
    # TODO: Allow negative WFE drift, but subtract delta WFE?
    assert wfe_drift>=0, '`wfe_drift` should not be negative' 
    wfe_drift_keys = _wfe_drift_key(self, coord_vals, coord_frame)
    ukeys = np.unique(wfe_drift_keys)
    nkeys = len(ukeys)
    if nkeys==1:
        psf_coeff_mod += _coeff_mod_wfe_drift(self, wfe_drift, key=ukeys[0])
    else:
        # Create coefficients for each unique key
        cf_mod_list = []
        for key in ukeys:
            cf_mod = _coeff_mod_wfe_drift(self, wfe_drift, key=key)
            cf_mod_list.append(cf_mod)
        # At each field position, find the corresponding coeff modification
        for i in range(nfield):
            ind = np.where(ukeys==wfe_drift_keys[i])[0][0]
            cf_mod = cf_mod_list[ind]
            psf_coeff_mod[i] += cf_mod

    # return psf_coeff, psf_coeff_mod

    # Add modifications to coefficients
    psf_coeff_mod += psf_coeff
    del psf_coeff
    psf_coeff = psf_coeff_mod

    # if multiple field points were present, we want to return PSF for each location
    if nfield>1:
        psf_all = []
        for ii in trange(nfield, leave=True, desc='PSFs'):
            # Just a single spectrum? Or unique spectrum at each field point?
            sp_norm = sp[ii] if((sp is not None) and (nspec==nfield)) else sp

            # Delete coefficients as they are used to reduce memory usage
            cf_ii = psf_coeff[ii]
            res = gen_image_from_coeff(self, cf_ii, psf_coeff_hdr, sp_norm=sp_norm,
                                       return_oversample=return_oversample)

            # For grisms (etc), the wavelength solution is the same for each field point
            wave, psf = res if is_spec else (None, res)
            psf_all.append(psf)

        if return_hdul:
            xvals, yvals = coord_vals
            hdul = fits.HDUList()
            for ii, psf in enumerate(psf_all):
                hdr = psf_coeff_hdr.copy()
                if return_oversample:
                    hdr['EXTNAME'] = ('OVERDIST', 'This extension is oversampled')
                    hdr['OSAMP'] = (self.oversample, 'Image oversampling rel to det')
                else:
                    hdr['EXTNAME'] = ('DET_DIST', 'This extension is detector sampled')
                    hdr['OSAMP'] = (1, 'Image oversampling rel to det')
                    hdr['PIXELSCL'] = self.pixelscale

                cunits = 'pixels' if ('sci' in coord_frame) or ('det' in coord_frame) else 'arcsec'
                hdr['XVAL']     = (xvals[ii], f'[{cunits}] Input X coordinate')
                hdr['YVAL']     = (yvals[ii], f'[{cunits}] Input Y coordinate')
                hdr['CFRAME']   = (coord_frame, 'Specified coordinate frame')
                hdr['WFEDRIFT'] = (wfe_drift, '[nm] WFE drift amplitude')
                hdul.append(fits.ImageHDU(data=psf, header=hdr))
            # Append wavelength solution
            if wave is not None:
                hdul.append(fits.ImageHDU(data=wave, name='Wavelengths'))
            output = hdul
        else:
            psf_all = np.asarray(psf_all)
            output = (wave, psf_all) if is_spec else psf_all
    else:
        res = gen_image_from_coeff(self, psf_coeff, psf_coeff_hdr, sp_norm=sp,
                                   return_oversample=return_oversample)

        if return_hdul:
            # For grisms (etc), the wavelength solution is the same for each field point
            wave, psf = res if is_spec else (None, res)

            hdr = psf_coeff_hdr.copy()
            if return_oversample:
                hdr['EXTNAME'] = ('OVERDIST', 'This extension is oversampled')
                hdr['OSAMP'] = (self.oversample, 'Image oversampling rel to det')
            else:
                hdr['EXTNAME'] = ('DET_DIST', 'This extension is detector sampled')
                hdr['OSAMP'] = (1, 'Image oversampling rel to det')
                hdr['PIXELSCL'] = self.pixelscale
                
            if coord_vals is not None:
                cunits = 'pixels' if ('sci' in coord_frame) or ('det' in coord_frame) else 'arcsec'
                hdr['XVAL']   = (coord_vals[0], f'[{cunits}] Input X coordinate')
                hdr['YVAL']   = (coord_vals[1], f'[{cunits}] Input Y coordinate')
                hdr['CFRAME'] = (coord_frame, 'Specified coordinate frame')
            else:
                cunits = 'pixels'
                hdr['XVAL']   = (hdr['DET_X'], f'[{cunits}] Input X coordinate')
                hdr['YVAL']   = (hdr['DET_Y'], f'[{cunits}] Input Y coordinate')
                hdr['CFRAME'] = ('sci', 'Specified coordinate frame')
            hdr['WFEDRIFT'] = (wfe_drift, '[nm] WFE drift amplitude')

            hdul = fits.HDUList()
            # Append each spectrum
            if nspec<=1:
                hdul.append(fits.ImageHDU(data=psf, header=hdr))
            else:
                for ii in range(nspec):
                    hdul.append(fits.ImageHDU(data=psf[ii], header=hdr))

            # Append wavelength solution
            if wave is not None:
                hdul.append(fits.ImageHDU(data=wave, name='Wavelengths'))
            output = hdul
        else:
            output = res
    
    return output

def _wfe_drift_key(self, coord_vals, coord_frame):
    """
    Choose which WFE drift coefficient key to use based on coordinate position.
    Only for coronagraphic observations (on-axis vs off-axis PSFs).
    """
    xsci = ysci = None

    # Preceed only if coronagraphic observation
    if not self.is_coron:
        return 'wfe_drift' 

    # Modify PSF coefficients based on position
    if coord_vals is None:
        return 'wfe_drift'
    else:
        # Determine sci coordinates
        cframe = coord_frame.lower()
        if cframe=='sci':
            xsci, ysci = coord_vals  # pixels
        elif cframe in ['det', 'tel', 'idl']:
            x = np.array(coord_vals[0])
            y = np.array(coord_vals[1])
            
            try:
                apname = self.aperturename
                siaf_ap = self.siaf[apname]
                xsci, ysci = siaf_ap.convert(x,y, cframe, 'sci')  
            except: 
                # apname = self.get_siaf_apname()
                self._update_aperturename()
                apname = self.aperturename
                if apname is None:
                    _log.warning('No suitable aperture name defined to determine (xsci,ysci) coordiantes')
                else:
                    _log.warning('self.siaf_ap not defined; assuming {}'.format(apname))
                    siaf_ap = self.siaf[apname]
                    xsci, ysci = siaf_ap.convert(x,y, cframe, 'sci')  # arcsec
                _log.warning('Update self.siaf_ap for more specific conversions to (xsci,ysci).')
        else:
            # Can't figure out coordinate frames, so assume 0
            return 'wfe_drift'

    # Assume we successfully found (xsci,ysci)
    if (xsci is not None):
        nfield = np.size(xsci)
        field_rot = 0 if self._rotation is None else self._rotation

        apname = self.aperturename
        siaf_ap = self.siaf[apname]

        # Convert to mask shifts (arcsec)
        xoff_asec = (xsci - siaf_ap.XSciRef) * siaf_ap.XSciScale
        yoff_asec = (ysci - siaf_ap.YSciRef) * siaf_ap.YSciScale
        xoff, yoff = xy_rot(-1*xoff_asec, -1*yoff_asec, field_rot)
        roff = np.sqrt(xoff**2 + yoff**2)

        # For bar masks, evaluate based on vertical distance
        if self.is_coron and self.image_mask[-1]=='B':
            roff = np.abs(yoff)

        if nfield==1:
            roff = [roff]

        # Choose either wfe_drift or wfe_drift_off for each position
        res = []
        for r in roff:
            key = 'wfe_drift' if r<0.1 else 'wfe_drift_off'
            res.append(key)

        return res
    else:
        return 'wfe_drift'


def _coeff_mod_wfe_drift(self, wfe_drift, key='wfe_drift'):
    """
    Modify PSF polynomial coefficients as a function of WFE drift.
    """

    # Modify PSF coefficients based on WFE drift
    if wfe_drift==0:
        cf_mod = 0 # Don't modify coefficients
    elif (self._psf_coeff_mod[key] is None):
        _log.warning("You must run `gen_wfedrift_coeff` first before setting the wfe_drift parameter.")
        _log.warning("Will continue assuming `wfe_drift=0`.")
        cf_mod = 0
    else:
        _log.info("Generating WFE drift modifications...")
        psf_coeff = self.psf_coeff

        cf_fit = self._psf_coeff_mod[key] 
        lxmap  = self._psf_coeff_mod['wfe_drift_lxmap'] 

        # Fit function
        cf_fit_shape = cf_fit.shape
        cf_fit = cf_fit.reshape([cf_fit.shape[0], -1])
        cf_mod = jl_poly(np.array([wfe_drift]), cf_fit, use_legendre=True, lxmap=lxmap)
        cf_mod = cf_mod.reshape(cf_fit_shape[1:])

        # Pad cf_mod array with 0s if undersized
        if not np.allclose(psf_coeff.shape, cf_mod.shape):
            new_shape = psf_coeff.shape[1:]
            cf_mod_resize = np.asarray([pad_or_cut_to_size(im, new_shape) for im in cf_mod])
            cf_mod = cf_mod_resize
    
    return cf_mod


def _coeff_mod_wfe_field(self, coord_vals, coord_frame, siaf_ap=None):
    """
    Modify PSF polynomial coefficients as a function of V2/V3 position.
    """

    v2 = v3 = None
    cf_mod = 0
    nfield = None

    psf_coeff_hdr = self.psf_coeff_header
    psf_coeff     = self.psf_coeff

    cf_fit = self._psf_coeff_mod['si_field'] 
    v2grid  = self._psf_coeff_mod['si_field_v2grid'] 
    v3grid  = self._psf_coeff_mod['si_field_v3grid']
    apname  = self.aperturename # self._psf_coeff_mod['si_field_apname']
    siaf_ap = self.siaf[apname] if apname is not None else None

    # Modify PSF coefficients based on position
    if coord_vals is None:
        pass
    elif self._psf_coeff_mod['si_field'] is None:
        si_wfe_str = 'True' if self.include_si_wfe else 'False'
        _log.info(f"Skipping WFE field dependence: self._psf_coeff_mod['si_field']=None and self.include_si_wfe={si_wfe_str}")
        # _log.warning("You must run `gen_wfefield_coeff` first before setting the coord_vals parameter.")
        # _log.warning("`calc_psf_from_coeff` will continue with default PSF.")
        cf_mod = 0
    else:
        si_field_apname = self._psf_coeff_mod.get('si_field_apname')
        siaf_ap_field = self.siaf[si_field_apname]

        # Assume cframe corresponds to siaf_ap input
        siaf_ap = siaf_ap_field if siaf_ap is None else siaf_ap
        cframe = coord_frame.lower()

        # Determine V2/V3 coordinates
        # Convert to common 'tel' coordinates
        if (siaf_ap.AperName != siaf_ap_field.AperName):
            x = np.array(coord_vals[0])
            y = np.array(coord_vals[1])
            v2, v3 = siaf_ap.convert(x,y, cframe, 'tel')
            v2, v3 = (v2/60., v3/60.) # convert to arcmin
        elif cframe=='tel':
            v2, v3 = coord_vals
            v2, v3 = (v2/60., v3/60.) # convert to arcmin
        elif cframe in ['det', 'sci', 'idl']:
            x = np.array(coord_vals[0])
            y = np.array(coord_vals[1])
            v2, v3 = siaf_ap.convert(x,y, cframe, 'tel')
            v2, v3 = (v2/60., v3/60.) # convert to arcmin
        else:
            _log.warning("coord_frame setting '{}' not recognized.".format(coord_frame))
            _log.warning("`calc_psf_from_coeff` will continue with default PSF.")

    # PSF Modifications assuming we successfully found v2/v3
    if (v2 is not None):
        _log.info("Generating field-dependent modifications...")
        # print(v2,v3)
        nfield = np.size(v2)
        cf_mod = field_coeff_func(v2grid, v3grid, cf_fit, v2, v3)
        cf_shape = psf_coeff.shape
        # cf_mod = np.zeros([nfield, cf_shape[0], cf_shape[1], cf_shape[2]])

        # Pad cf_mod array with 0s if undersized
        psf_cf_dim = len(cf_shape)
        if not np.allclose(cf_shape, cf_mod.shape[-psf_cf_dim:]):
            new_shape = cf_shape[1:]
            cf_mod_resize = np.asarray([pad_or_cut_to_size(im, new_shape) for im in cf_mod])
            del cf_mod
            cf_mod = cf_mod_resize

    return cf_mod, nfield

def _coeff_mod_wfe_mask(self, coord_vals, coord_frame, siaf_ap=None):
    """
    Modify PSF polynomial coefficients as a function of V2/V3 position.

    Parameters
    ----------
    coord_vals : tuple or None
        Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
    coord_frame : str
        Type of desired output coordinates. 

            * 'tel': arcsecs V2,V3
            * 'sci': pixels, in conventional DMS axes orientation
            * 'det': pixels, in raw detector read out axes orientation
            * 'idl': arcsecs relative to aperture reference location.
    """

    xidl = yidl = None
    cf_mod = 0
    nfield = None

    psf_coeff_hdr = self.psf_coeff_header
    psf_coeff     = self.psf_coeff

    cf_fit = self._psf_coeff_mod.get('si_mask', None) 

    # Coord values are set, but not coefficients supplied
    if (coord_vals is not None) and (cf_fit is None):
        _log.warning("You must run `gen_wfemask_coeff` first before setting the coord_vals parameter for masked focal planes.")
        _log.info("`calc_psf_from_coeff` will continue without mask field dependency.")
    # No coord values, but NIRCam bar/wedge mask in place
    elif (coord_vals is None) and (self.name=='NIRCam') and (self.image_mask[-1]=='B'):
        # Determine desired location along bar
        bar_offset = self.get_bar_offset()
        if (bar_offset != 0) and (cf_fit is None):
            _log.warning("You must run `gen_wfemask_coeff` to obtain PSFs offset along bar mask.")
            _log.info("`calc_psf_from_coeff` will continue assuming bar_offset=0.")
        else:
            nfield = 1
            # Get coords in arcsec
            xidl = bar_offset
            yidl = 0
            # # Get sci coords in pixels
            # bar_offset_pix = bar_offset / siaf_ap.XSciScale
            # xsci = siaf_ap.XSciRef + bar_offset_pix
            # ysci = siaf_ap.YSciRef
    elif (coord_vals is not None):
        si_mask_apname = self._psf_coeff_mod.get('si_mask_apname')
        siaf_ap_mask = self.siaf[si_mask_apname]

        # Assume cframe corresponds to siaf_ap input
        siaf_ap = siaf_ap_mask if siaf_ap is None else siaf_ap
        cframe = coord_frame.lower()

        # Convert to common 'tel' coordinates
        if (siaf_ap.AperName != siaf_ap_mask.AperName):
            x = np.array(coord_vals[0])
            y = np.array(coord_vals[1])
            xtel, ytel = siaf_ap.convert(x,y, cframe, 'tel')
            xidl, yidl = siaf_ap_mask.convert(xtel, ytel, 'tel', 'idl')
        elif cframe=='idl':
            xidl, yidl = coord_vals
        elif cframe in ['det', 'tel', 'sci']:
            x = np.array(coord_vals[0])
            y = np.array(coord_vals[1])
            xidl, yidl = siaf_ap.convert(x,y, cframe, 'idl')
        else:
            _log.warning("coord_frame setting '{}' not recognized.".format(coord_frame))
            _log.warning("`calc_psf_from_coeff` will continue with default PSF.")

    # PSF Modifications assuming we successfully found (xsci,ysci)
    # print(xidl, yidl)
    if (xidl is not None):
        _log.info("Generating mask-dependent modifications...")
        # print(v2,v3)
        nfield = np.size(xidl)
        field_rot = 0 if self._rotation is None else self._rotation

        # Convert to mask shifts (arcsec)
        xoff_asec, yoff_asec = (xidl, yidl)
        xoff_cf, yoff_cf = xy_rot(-1*xoff_asec, -1*yoff_asec, field_rot)


        if (self.name=='NIRCam') and (np.any(np.abs(xoff_asec)>12) or np.any(np.abs(yoff_asec)>12)):
            _log.warn("Some values outside mask FoV (beyond 12 asec offset)!")
            
        # print(xoff_asec, yoff_asec)
        xgrid  = self._psf_coeff_mod['si_mask_xgrid']  # arcsec
        ygrid  = self._psf_coeff_mod['si_mask_ygrid']  # arcsec
        cf_mod = field_coeff_func(xgrid, ygrid, cf_fit, xoff_cf, yoff_cf)

        # Pad cf_mod array with 0s if undersized
        psf_cf_dim = len(psf_coeff.shape)
        if not np.allclose(psf_coeff.shape, cf_mod.shape[-psf_cf_dim:]):
            new_shape = psf_coeff.shape[1:]
            cf_mod_resize = np.asarray([pad_or_cut_to_size(im, new_shape) for im in cf_mod])
            cf_mod = cf_mod_resize

    return cf_mod, nfield


def coron_grid(self, npsf_per_axis, xoff_vals=None, yoff_vals=None):
    """Get grid points based on coronagraphic obseervation
    
    Returns sci pixels values around mask center.
    """
    
    def log_grid(nvals, vmax=10):
        """Log spacing in arcsec relative to mask center"""
        # vals_p = np.logspace(-2,np.log10(vmax),int((nvals-1)/2))
        vals_p = np.geomspace(0.01, vmax, int((nvals-1)/2))
        vals_m = np.sort(-1*vals_p)
        return np.sort(np.concatenate([vals_m, [0], vals_p]))

    def lin_grid(nvals, vmin=-10, vmax=10):
        """Linear spacing in arcsec relative to mask center"""
        return np.linspace(vmin, vmax, nvals)

    # Observation aperture
    siaf_ap = self.siaf[self.aperturename]
    
    if self.name.lower()=='nircam':
        nx_pix = 300 if self.channel.lower()=='long' else 600
        ny_pix = nx_pix
    else:
        xvert, yvert = siaf_ap.corners('sci', rederive=False)
        xsci_min, xsci_max = int(np.min(xvert)), int(np.max(xvert))
        ysci_min, ysci_max = int(np.min(yvert)), int(np.max(yvert))

        nx_pix = int(xsci_max - xsci_min)
        ny_pix = int(ysci_max - ysci_min)

    xoff_min, xoff_max = self.pixelscale * np.array([-1,1]) * nx_pix / 2
    yoff_min, yoff_max = self.pixelscale * np.array([-1,1]) * ny_pix / 2
        
    if np.size(npsf_per_axis)==1:
        xpsf = ypsf = npsf_per_axis
    else:
        xpsf, ypsf = npsf_per_axis

    field_rot = 0 if self._rotation is None else self._rotation
    xlog_spacing = False if self.image_mask[-2:]=='WB' else True
    ylog_spacing = True 
    
    if xoff_vals is None:
        xmax = np.abs([xoff_min,xoff_max]).max()
        xoff = log_grid(xpsf, xmax) if xlog_spacing else lin_grid(xpsf, -xmax, xmax)
    else:
        xoff = xoff_vals
    if yoff_vals is None:
        ymax = np.abs([yoff_min,yoff_max]).max()
        yoff = log_grid(ypsf, ymax) if ylog_spacing else lin_grid(ypsf, -ymax, ymax)
    else:
        yoff = yoff_vals

    # Mask Offset grid positions in arcsec
    xgrid_off, ygrid_off = np.meshgrid(xoff, yoff)
    xgrid_off, ygrid_off = xgrid_off.flatten(), ygrid_off.flatten()

    # Offsets relative to center of mask
    xoff_asec, yoff_asec = xy_rot(-1*xgrid_off, -1*ygrid_off, -1*field_rot)
    xtel, ytel = siaf_ap.convert(xoff_asec, yoff_asec, 'idl', 'tel')

    # Convert from aperture used to create mask into sci pixels for observe aperture
    xsci, ysci = self.siaf_ap.convert(xtel, ytel, 'tel', 'sci')

    return xsci, ysci

def _calc_psfs_grid(self, sp=None, wfe_drift=0, osamp=1, npsf_per_full_fov=15,
                    xsci_vals=None, ysci_vals=None, return_coords=None,
                    use_coeff=True, **kwargs):

    """Create a grid of PSFs across an instrumnet FoV
    
    Create a grid of PSFs across instrument aperture FoV. By default,
    imaging observations will be for full detector FoV with regularly
    spaced grid. Coronagraphic observations will cover nominal 
    coronagraphic mask region (usually 10s of arcsec) and will have
    logarithmically spaced values.

    Keyword Args
    ============
    sp : :mod:`pysynphot.spectrum`
        If not specified, the default is flat in phot lam (equal number of photons 
        per wavelength bin). The default is normalized to produce 1 count/sec within 
        that bandpass, assuming the telescope collecting area and instrument bandpass. 
        Coronagraphic PSFs will further decrease this due to the smaller pupil
        size and suppression of coronagraphic mask. 
        If set, then the resulting PSF image will be scaled to generate the total
        observed number of photons from the spectrum (ie., not scaled by unit response).
    wfe_drift : float
        Desired WFE drift value relative to default OPD.
    osamp : int
        Sampling of output PSF relative to detector sampling.
    npsf_per_full_fov : int
        Number of PSFs across one dimension of the instrument's field of 
        view. If a coronagraphic observation, then this is for the nominal
        coronagrahic field of view.
    xsci_vals: None or ndarray
        Option to pass a custom grid values along x-axis in 'sci' coords.
        If coronagraph, this instead corresponds to coronagraphic mask axis in arcsec, 
        which has a slight rotation relative to detector axis in MIRI.
    ysci_vals: None or ndarray
        Option to pass a custom grid values along y-axis in 'sci' coords.
        If coronagraph, this instead corresponds to coronagraphic mask axis in arcsec, 
        which has a slight rotation relative to detector axis in MIRI.
    return_coords : None or str
        Option to also return coordinate values in desired frame 
        ('det', 'sci', 'tel', 'idl').
        Output is then xvals, yvals, hdul_psfs.
    use_coeff : bool
        If True, uses `calc_psf_from_coeff`, other WebbPSF's built-in `calc_psf`.
    """

    # Observation aperture
    siaf_ap_obs = self.siaf_ap

    # Produce grid of PSF locations across the field of view
    if self.is_coron:
        xsci_psf, ysci_psf = coron_grid(self, npsf_per_full_fov, 
                                        xoff_vals=xsci_vals, yoff_vals=ysci_vals)
    else:
        # No need to go beyond detector pixels

        # Number of sci pixels in FoV
        # Generate grid borders
        xvert, yvert = siaf_ap_obs.closed_polygon_points('sci', rederive=False)
        xsci_min, xsci_max = int(np.min(xvert)), int(np.max(xvert))
        ysci_min, ysci_max = int(np.min(yvert)), int(np.max(yvert))

        nx_pix = int(xsci_max - xsci_min)
        ny_pix = int(ysci_max - ysci_min)

        # Ensure at least 5 PSFs across FoV for imaging
        if np.size(npsf_per_full_fov)==1:
            xpsf_full = ypsf_full = npsf_per_full_fov
        else:
            xpsf_full, ypsf_full = npsf_per_full_fov

        xpsf = np.max([int(xpsf_full * nx_pix / siaf_ap_obs.XDetSize), 5])
        ypsf = np.max([int(ypsf_full * ny_pix / siaf_ap_obs.YDetSize), 5])
        # Cut in half for NIRCam SW (4 detectors per FoV)
        if self.name.lower()=='nircam' and self.channel.lower()=='short':
            xpsf = np.max([int(xpsf / 2), 5])
            ypsf = np.max([int(ypsf / 2), 5])

        # Create linear set of grid points along x and y axes 
        if xsci_vals is None:
            xsci_vals = np.linspace(xsci_min, xsci_max, xpsf)
        if ysci_vals is None:
            ysci_vals = np.linspace(ysci_min, ysci_max, ypsf)

        # Full set of grid points to generate PSFs
        xsci_psf, ysci_psf = np.meshgrid(xsci_vals, ysci_vals)
        xsci_psf = xsci_psf.flatten()
        ysci_psf = ysci_psf.flatten()

    # Convert everything to tel for good measure to store in header
    xtel_psf, ytel_psf = siaf_ap_obs.convert(xsci_psf, ysci_psf, 'sci', 'tel')

    if use_coeff:
        hdul_psfs = self.calc_psf_from_coeff(sp=sp, coord_vals=(xtel_psf, ytel_psf), coord_frame='tel', 
                                             wfe_drift=wfe_drift, return_oversample=True, **kwargs)
    else:
        hdul_psfs = fits.HDUList()
        npos = len(xtel_psf)
        for xoff, yoff in tqdm(zip(xtel_psf, ytel_psf), total=npos):
            res = self.calc_psf(sp=sp, coord_vals=(xoff,yoff), coord_frame='tel', 
                                return_oversample=True)
            # If add_distortion take index 2, otherwise index 0
            hdu = res[2] if len(res)==4 else res[0]
            hdul_psfs.append(hdu)

    # Resample if necessary
    scale = osamp / self.oversample #hdu.header['OSAMP']
    if scale != 1:
        for hdu in hdul_psfs:
            hdu.data = frebin(hdu.data, scale=scale)
            hdu.header['PIXELSCL'] = hdu.header['PIXELSCL'] / scale
            hdu.header['OSAMP'] = osamp

    if return_coords is None:
        return hdul_psfs
    elif return_coords=='sci':
        xvals, yvals = xsci_psf, ysci_psf
    elif return_coords=='tel':
        xvals, yvals = xtel_psf, ytel_psf
    else:
        xvals, yvals = siaf_ap_obs.convert(xsci_psf, ysci_psf, 'sci', return_coords)
        
    return xvals, yvals, hdul_psfs


def _calc_psfs_sgd(self, xoff_asec, yoff_asec, use_coeff=True, return_oversample=True, **kwargs):
    """Calculate small grid dithers PSFs"""

    if self.is_coron==False:
        _log.warn("`calc_sgd` only valid for coronagraphic observations (set `image_mask` attribute).")
        return

    if use_coeff:
        result = self.calc_psf_from_coeff(coord_frame='idl', coord_vals=(xoff_asec,yoff_asec), 
                                          return_oversample=return_oversample, siaf_ap=self.siaf_ap, **kwargs)
    else:
        log_prev = conf.logging_level
        setup_logging('WARN', verbose=False)

        npos = len(xoff_asec)
        # Return HDUList or array of images?
        if kwargs.get('return_hdul',True):
            result = fits.HDUList()
            for xoff, yoff in tqdm(zip(xoff_asec, yoff_asec), total=npos):
                res = self.calc_psf(coord_frame='idl', coord_vals=(xoff,yoff), 
                                    return_oversample=return_oversample, **kwargs)
                if len(res)==4:
                    hdu = res[2] if return_oversample else res[3]
                else:
                    hdu = res[0] if return_oversample else res[1]
                result.append(hdu)
        else:
            result = []
            for xoff, yoff in tqdm(zip(xoff_asec, yoff_asec), total=npos):
                res = self.calc_psf(coord_frame='idl', coord_vals=(xoff,yoff), 
                                    return_oversample=return_oversample, **kwargs)
                result.append(res)
            result = np.asarray(result)

        setup_logging(log_prev, verbose=False)

    return result

def nrc_mask_trans(image_mask, x, y):
    """ Compute the amplitude transmission appropriate for a BLC for some given pixel spacing
    corresponding to the supplied Wavefront.

    Based on the Krist et al. SPIE paper on NIRCam coronagraph design

    *Note* : To get the actual transmission, these values should be squared.

    """

    import scipy

    if not isinstance(x, np.ndarray):
        x = np.asarray([x]).flatten()
        y = np.asarray([y]).flatten()

    if image_mask[-1]=='R':

        r = poppy.accel_math._r(x, y)
        if image_mask == 'MASK210R':
            sigma = 5.253
        elif image_mask == 'MASK335R':
            sigma = 3.2927866
        elif image_mask == 'MASK430R':
            sigma = 2.58832

        sigmar = sigma * r

        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        bessel_j1_zero2 = scipy.special.jn_zeros(1, 2)[1]
        sigmar.clip(np.finfo(sigmar.dtype).tiny, bessel_j1_zero2, out=sigmar)  # avoid divide by zero -> NaNs
        transmission = (1 - (2 * scipy.special.j1(sigmar) / sigmar) ** 2)
        transmission[r == 0] = 0  # special case center point (value based on L'Hopital's rule)

    if image_mask[-1]=='B':
        # This is hard-coded to the wedge-plus-flat-regions shape for NIRCAM

        # the scale fact should depend on X coord in arcsec, scaling across a 20 arcsec FOV.
        # map flat regions to 2.5 arcsec each
        # map -7.5 to 2, +7.5 to 6. slope is 4/15, offset is +9.5
        wedgesign = 1 if image_mask == 'MASKSWB' else -1  # wide ends opposite for SW and LW

        scalefact = (2 + (x * wedgesign + 7.5) * 4 / 15).clip(2, 6)

        # Working out the sigma parameter vs. wavelength to get that wedge pattern is non trivial
        # This is NOT a linear relationship. See calc_blc_wedge helper fn below.

        if image_mask == 'MASKSWB':  
            polyfitcoeffs = np.array([2.01210737e-04, -7.18758337e-03, 1.12381516e-01,
                                        -1.00877701e+00, 5.72538509e+00, -2.12943497e+01,
                                        5.18745152e+01, -7.97815606e+01, 7.02728734e+01])
        elif image_mask == 'MASKLWB':  
            polyfitcoeffs = np.array([9.16195583e-05, -3.27354831e-03, 5.11960734e-02,
                                        -4.59674047e-01, 2.60963397e+00, -9.70881273e+00,
                                        2.36585911e+01, -3.63978587e+01, 3.20703511e+01])
        else:
            raise NotImplementedError(f"{image_mask} not a valid name for NIRCam wedge occulter")

        sigmas = scipy.poly1d(polyfitcoeffs)(scalefact)

        sigmar = sigmas * np.abs(y)
        # clip sigma: The minimum is to avoid divide by zero
        #             the maximum truncates after the first sidelobe to match the hardware
        sigmar.clip(min=np.finfo(sigmar.dtype).tiny, max=2 * np.pi, out=sigmar)
        transmission = (1 - (np.sin(sigmar) / sigmar) ** 2)
        transmission[y == 0] = 0 

        transmission[np.abs(x) > 10] = 1.0

    # Amplitude transmission (square to get intensity transmission; ie., photon throughput)
    return transmission


def _transmission_map(self, coord_vals, coord_frame, siaf_ap=None):

    if not self.is_coron:
        return None

    apname_mask = self._psf_coeff_mod['si_mask_apname']
    if apname_mask is None:
        apname_mask = self.aperturename
    siaf_ap_mask = self.siaf[apname_mask]

    # Assume cframe corresponds to siaf_ap input
    siaf_ap = siaf_ap_mask if siaf_ap is None else siaf_ap
    coord_frame = coord_frame.lower()

    # Convert to common 'tel' coordinates
    cx, cy = np.asarray(coord_vals)
    if (siaf_ap.AperName != siaf_ap_mask.AperName):
        cx_tel, cy_tel = siaf_ap.convert(cx, cy, coord_frame, 'tel')
        cx_idl, cy_idl = siaf_ap_mask.convert(cx_tel, cy_tel, 'tel', 'idl')
    elif coord_frame=='idl':
        cx_idl, cy_idl = (cx, cy)
    elif coord_frame in ['det', 'tel', 'sci']:
        cx_idl, cy_idl = siaf_ap_mask.convert(cx, cy, coord_frame, 'idl')

    # Get mask transmission
    trans = nrc_mask_trans(self.image_mask, cx_idl, cy_idl)

    return trans, cx_idl, cy_idl

def _nrc_coron_psf_sums(self, coord_vals, coord_frame, siaf_ap=None, return_max=False, trans=None):
    """
    Function to analytically determine the sum and max value 
    of a NIRCam off-axis coronagraphic PSF while partially
    occulted by the coronagrpahic mask.

    Keyword Args
    ============
    return_max : bool
        By default, this function returns the PSF sums. Set this keyword
        to True in order to return the PSF max values instead.
    trans : float, ndarray, or None
        Transmission is usually sampled from mask transmission function
        corresponding to input coordinates. Instead, supply the transmission
        value directly. 
    """

    # Ensure correct scaling for off-axis PSFs
    if not self.is_coron:
        return None

    # Get mask transmission
    t_temp, cx_idl, cy_idl = _transmission_map(self, coord_vals, coord_frame, siaf_ap=siaf_ap)
    if trans is None:
        trans = t_temp**2

    # Linear combination of min/max to determine PSF sum
    # Get a and b values for each position
    avals = trans
    bvals = 1 - avals

    # print(avals, bvals)

    # Store PSF sums for later retrieval
    try:
        psf_sums_dict = self._psf_sums
    except AttributeError:
        psf_sums_dict = {}
        self._psf_sums = psf_sums_dict

    # Offset PSF sum
    psf_off_sum = psf_sums_dict.get('psf_off', None)
    psf_off_max = psf_sums_dict.get('psf_off_max', None)
    if (psf_off_sum is None) or (psf_off_max is None):
        psf = _calc_psf_from_coeff(self, return_oversample=False, return_hdul=False, 
                                   coord_vals=(10,10), coord_frame='idl')
        psf_off_sum = psf.sum()
        psf_off_max = np.max(pad_or_cut_to_size(psf,10))
        psf_sums_dict['psf_off'] = psf_off_sum
        psf_sums_dict['psf_off_max'] = psf_off_max

    # Central PSF sum(s)
    if self.image_mask[-1] == 'R':
        psf_cen_sum = psf_sums_dict.get('psf_cen', None)
        psf_cen_max = psf_sums_dict.get('psf_cen_max', None)
        if (psf_cen_sum is None) or (psf_cen_max is None):
            psf = _calc_psf_from_coeff(self, return_oversample=False, return_hdul=False)
            psf_cen_sum = psf.sum()
            psf_sums_dict['psf_cen'] = psf_cen_sum
            psf_sums_dict['psf_cen_max'] = np.max(pad_or_cut_to_size(psf,10))
    elif self.image_mask[-1] == 'B':
        # Build a list of bar offsets to interpolate scale factors
        xvals = psf_sums_dict.get('psf_cen_xvals', None)
        psf_cen_sum_arr = psf_sums_dict.get('psf_cen_sum_arr', None)
        psf_cen_max_arr = psf_sums_dict.get('psf_cen_max_arr', None)

        if (psf_cen_sum_arr is None) or (psf_cen_max_arr is None):
            xvals = np.linspace(-8,8,9)
            psf_sums_dict['psf_cen_xvals'] = xvals

            psf_cen_sum_arr = []
            psf_cen_max_arr = []
            for xv in xvals:
                psf= _calc_psf_from_coeff(self, return_oversample=False, return_hdul=False, 
                                          coord_vals=(xv,0), coord_frame='idl')
                psf_cen_sum_arr.append(psf.sum())
                psf_cen_max_arr.append(np.max(pad_or_cut_to_size(psf,10)))
            psf_cen_sum_arr = np.array(psf_cen_sum_arr)
            psf_cen_max_arr = np.array(psf_cen_max_arr)
            psf_sums_dict['psf_cen_sum_arr'] = psf_cen_sum_arr
            psf_sums_dict['psf_cen_max_arr'] = psf_cen_max_arr

        # Interpolation function
        finterp = interp1d(xvals, psf_cen_sum_arr, kind='linear', fill_value='extrapolate')
        psf_cen_sum = finterp(cx_idl)
        finterp = interp1d(xvals, psf_cen_max_arr, kind='linear', fill_value='extrapolate')
        psf_cen_max = finterp(cx_idl)
    else:
        _log.warn(f"Image mask not recognized: {self.image_mask}")
        return None
        
    if return_max:
        psf_max = avals * psf_off_max + bvals * psf_cen_max
        return psf_max
    else:
        psf_sum = avals * psf_off_sum + bvals * psf_cen_sum
        return psf_sum


def _nrc_coron_rescale(self, res, coord_vals, coord_frame, siaf_ap=None, sp=None):
    """
    Function for better scaling of NIRCam coronagraphic output for sources
    that overlap the image masks. 
    """

    if coord_vals is None:
        return res

    nfield = np.size(coord_vals[0])
    psf_sum = _nrc_coron_psf_sums(self, coord_vals, coord_frame, siaf_ap=siaf_ap)
    if psf_sum is None:
        return res

    # Scale by countrate of observed spectrum
    if (sp is not None) and (not isinstance(sp, list)):
        nspec = 1
        obs = S.Observation(sp, self.bandpass, binset=self.bandpass.wave)
        sp_counts = obs.countrate()
    elif (sp is not None) and (isinstance(sp, list)):
        nspec = len(sp)
        if nspec==1:
            obs = S.Observation(sp[0], self.bandpass, binset=self.bandpass.wave)
            sp_counts = obs.countrate()
        else:
            sp_counts = []
            for i, sp_norm in enumerate(sp):
                obs = S.Observation(sp_norm, self.bandpass, binset=self.bandpass.wave)
                sp_counts.append(obs.countrate())
            sp_counts = np.array(sp_counts)
    else:
        nspec = 0
        sp_counts = 1

    if nspec>1 and nspec!=nfield:
        _log.warn("Number of spectra should be 1 or equal number of field points")

    # Scale by count rate
    psf_sum *= sp_counts

    # Re-scale PSF by total sums
    if isinstance(res, fits.HDUList):
        for i, hdu in enumerate(res):
            hdu.data *= (psf_sum[i] / hdu.data.sum())
    elif nfield==1:
        res *= (psf_sum[0] / res.sum())
    else:
        for i, data in enumerate(res):
            data *= (psf_sum[i] / data.sum())

    return res


