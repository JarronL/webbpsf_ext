# Import libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import datetime, time
import os, sys, platform, six
import multiprocessing as mp
import traceback

from astropy.io import fits, ascii
from astropy.table import Table
import astropy.units as u

from copy import deepcopy

from .utils import conf, webbpsf, poppy, setup_logging
from .bandpasses import miri_filter, nircam_filter
from .opds import OPDFile_to_HDUList
from .psfs import nproc_use, _wrap_coeff_for_mp, gen_image_from_coeff
from .psfs import make_coeff_resid_grid, field_coeff_func
from .maths import jl_poly, jl_poly_fit
from .image_manip import pad_or_cut_to_size
from .coords import NIRCam_V2V3_limits
from .version import __version__

import logging
_log = logging.getLogger('webbpsf_ext')

# from scipy.interpolate import griddata, RegularGridInterpolator, interp1d
from numpy.polynomial import legendre

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
    autogen : bool
        Option to automatically generate all PSF coefficients upon
        initialization. Otherwise, these need to be generated manually
        with `gen_psf_coeff`, gen_wfedrift_coeff`, and `gen_wfefield_coeff`.
        Default: False.
    """

    def __init__(self, filter=None, pupil_mask=None, image_mask=None, 
                 fov_pix=None, oversample=None, auto_gen_coeffs=False):
        
        webbpsf_NIRCam.__init__(self)

        # Initialize script
        _init_inst(self, filter=filter, pupil_mask=pupil_mask, image_mask=image_mask,
                   fov_pix=fov_pix, oversample=oversample)

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

        # Option to calculate ND acquisition for coronagraphic obs
        self._ND_acq = False

        if auto_gen_coeffs:
            self.gen_psf_coeff()
            self.gen_wfedrift_coeff()
            self.gen_wfefield_coeff()

    @property
    def save_dir(self):
        """Coefficient save directory"""
        # Default name
        if self._save_dir is None:
            # Name to save array of oversampled coefficients
            inst_name = self.name
            save_dir = conf.WEBBPSF_EXT_PATH + f'{inst_name}/psf_coeffs/'
            # Create directory if it doesn't already exist
            if not os.path.isdir(save_dir):
               os.makedirs(save_dir)
        else:
            save_dir = self._save_dir
        return save_dir
    @save_dir.setter
    def save_dir(self, value):
        self._save_dir = value

    @property
    def save_name(self):
        """Coefficient file name"""
        save_name = self._save_name
        save_name = self.gen_save_name() if save_name is None else save_name
        return save_name
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
    def ND_acq(self):
        """Use Coronagraphic ND acquisition square?"""
        return self._ND_acq
    @ND_acq.setter
    def ND_acq(self, value):
        """Set whether or not we're placed on an ND acquisition square."""
        _check_list(value, [True, False], 'ND_acq')
        self._ND_acq = value

    @property
    def fov_pix(self):
        return self._fov_pix
    @fov_pix.setter
    def fov_pix(self, value):
        self._fov_pix = value
        
    @property
    def oversample(self):
        if self._oversample is None:
            oversample = 2 if self.is_lyot else 4
        else:
            oversample = self._oversample
        return oversample
    @oversample.setter
    def oversample(self, value):
        self._oversample = value
    
    @property
    def npsf(self):
        
        npsf = self._npsf
        
        # Default to 10 PSF simulations per um
        w1 = self.bandpass.wave.min() / 1e4
        w2 = self.bandpass.wave.max() / 1e4
        if npsf is None:
            dn = 20 
            npsf = int(np.ceil(dn * (w2-w1)))

        # Want at least 5 monochromatic PSFs
        npsf = 5 if npsf<5 else int(npsf)

        # Number of points must be greater than degree of fit
        npsf = self.ndeg+1 if npsf<=self.ndeg else int(npsf)

        return npsf
    
    @npsf.setter
    def npsf(self, value):
        self._npsf = value
        
    @property
    def ndeg(self):
        ndeg = self._ndeg
        if ndeg is None:
            # TODO: Quantify these better
            if self.use_legendre:
                ndeg = 7 if self.quick else 9
            else:
                ndeg = 7 if self.quick else 9
        return ndeg
    @ndeg.setter
    def ndeg(self, value):
        self._ndeg = value
    @property
    def quick(self):
        """Perform quicker coeff calculation over limited bandwidth?"""
        return False if self._quick is None else self._quick
    @quick.setter
    def quick(self, value):
        """Perform quicker coeff calculation over limited bandwidth?"""
        _check_list(value, [True, False], 'quick')
        self._quick = value

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
            x, y = map(int, position)
        except ValueError:
            raise ValueError("Detector pixel coordinates must be a pair of numbers, not {}".format(position))
        self._detector_position = (int(position[0]), int(position[1]))

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
        bp = nircam_filter(self.filter, pupil=self.pupil_mask, mask=self.image_mask,
                           module=self.module, ND_acq=self.ND_acq, 
                           grism_order=self._grism_order)
        return bp

    def _addAdditionalOptics(self, optsys, oversample=2):
        """
        Add coronagraphic optics for NIRCam
        """
        # Allow arbitrary offsets of the focal plane masks with respect to the pixel grid origin;
        # In most use cases it's better to offset the star away from the mask instead, using
        # options['source_offset_*'], but doing it this way instead is helpful when generating
        # the Pandeia ETC reference PSF library.

        # mask_shift_x/y override coron_shift_x/y
        shift_x = self.options.get('mask_shift_x', None)
        shift_y = self.options.get('mask_shift_y', None)

        # This will apply mask shifts to the default coron_shift keys
        if shift_x is not None:
            self.options['coron_shift_x'] = shift_x
        if shift_y is not None:
            self.options['coron_shift_y'] = shift_y

        return super(NIRCam_ext, self)._addAdditionalOptics(optsys, oversample=oversample)

    def gen_mask_image(self, npix=None, pixelscale=None):
        """
        Return an image representation of the focal plane mask.
        Output is in 'sci' coords orientation.
        
        Parameters
        ==========
        npix : int
            Number of pixels in output image. If not set, then
            is automatically determined based on mask FoV and
            `pixelscale`
        pixelscale : float
            Size of output pixels in units of arcsec. If not specified,
            then selects nominal detector pixel scale.
        """
        
        from webbpsf.optics import NIRCam_BandLimitedCoron

        # mask_shift_x/y override coron_shift_x/y
        shift_x = self.options.get('mask_shift_x', None)
        shift_y = self.options.get('mask_shift_y', None)

        # This will apply mask shifts to the default coron_shift keys
        if shift_x is not None:
            self.options['coron_shift_x'] = shift_x
        if shift_y is not None:
            self.options['coron_shift_y'] = shift_y

        shifts = {'shift_x': self.options.get('coron_shift_x', None),
                  'shift_y': self.options.get('coron_shift_y', None)}

        if pixelscale is None:
            pixelscale = self.pixelscale / self.oversample
        npix = int(20 / pixelscale + 0.5) if npix is None else npix

        if self.is_coron:
            if ('MASK' in self.image_mask) and (self.image_mask[-1] == 'R'):
                bar_offset = None
                auto_offset = None
            else:
                bar_offset = self.options.get('bar_offset', None)
                if bar_offset is None:
                    auto_offset = self.filter
                else:
                    try:
                        _ = float(bar_offset)
                        auto_offset = None
                    except ValueError:
                        # If the "bar_offset" isn't a float, pass it to auto_offset instead
                        auto_offset = bar_offset
                        bar_offset = None
            mask = NIRCam_BandLimitedCoron(name=self.image_mask, module=self.module, 
                                           bar_offset=bar_offset, auto_offset=auto_offset, **shifts)

            wavelength = self.bandpass.avgwave() / 1e10

            wave = poppy.Wavefront(wavelength=wavelength, npix=npix, pixelscale=pixelscale)
            im = mask.get_transmission(wave)
        else:
            im = np.ones([npix,npix])

        return im

    def get_opd_info(self, opd=None, HDUL_to_OTELM=True):
        """
        Parse out OPD information for a given OPD, which can be a file name, tuple (file,slice), 
        HDUList, or OTE Linear Model. Returns dictionary of some relevant information for 
        logging purposes. The dictionary has an OPD version as an OTE LM.
        
        This outputs an OTE Linear Model. In order to update instrument class:
            >>> opd_dict = inst.get_opd_info()
            >>> opd_new = opd_dict['pupilopd']
            >>> inst.pupilopd = opd_new
            >>> inst.pupil = opd_new
        """
        return _get_opd_info(self, opd=opd, HDUL_to_OTELM=HDUL_to_OTELM)

    def drift_opd(self, wfe_drift, opd=None):
        """
        A quick method to drift the pupil OPD. This function applies some WFE drift to input 
        OPD file by breaking up the wfe_drift attribute into thermal, frill, and IEC components. 
        If we want more realistic time evolution, then we should use the procedure in 
        dev_utils/WebbPSF_OTE_LM.ipynb to create a time series of OPD maps, which can then be 
        passed directly to create unique PSFs.
        
        This outputs an OTE Linear Model. In order to update instrument class:
            >>> opd_new = inst.drift_opd()
            >>> inst.pupilopd = opd_new
            >>> inst.pupil = opd_new
        """
        return _drift_opd(self, wfe_drift, opd=opd)

    def gen_save_name(self, wfe_drift=0):
        """
        Generate save name for polynomial coefficient output file.
        """
        return _gen_save_name(self, wfe_drift=wfe_drift)

    def gen_psf_coeff(self, wfe_drift=0, nproc=None, force=False, 
                      save=True, return_results=False, return_extras=False):
        """Generate PSF coefficients

        Creates a set of coefficients that will generate a simulated PSF at any
        arbitrary wavelength. This function first uses simulates a number of
        evenly spaced PSFs throughout some specified bandpass. An nth-degree 
        polynomial is then fit to each oversampled pixel using a linear-least 
        squares fitting routine. The final set of coefficients for each pixel 
        is returned as an image cube. The returned set of coefficient can be 
        used to produce a set of PSFs by:

        >>> psfs = jl_poly(waves, coeffs)

        where 'waves' can be a scalar, nparray, list, or tuple. All wavelengths
        are in microns.

        Parameters
        ----------
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
            used for successive to determine varying WFE drift or focal plane dependencies.
        return_extras : bool
            Additionally returns a dictionary of monochromatic PSFs images and their 
            corresponding wavelengths for debugging purposes. Can be used with or without
            `return_results`. 
        """
        
        return _gen_psf_coeff(self, nproc=nproc, wfe_drift=wfe_drift, force=force, save=save, 
                              return_results=return_results, return_extras=return_extras)

    def gen_wfedrift_coeff(self, force=False, save=True, **kwargs):
        """ Fit WFE drift coefficients

        This function finds a relationship between PSF coefficients in the presence of WFE drift. 
        For a series of WFE drift values, we generate corresponding PSF coefficients and fit a 
        polynomial relationship to the residual values. This allows us to quickly modify a nominal 
        set of PSF image coefficients to generate a new PSF where the WFE has drifted by some amplitude.
        
        It's Legendre's all the way down...

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
        return _gen_wfedrift_coeff(self, force=force, save=save, **kwargs)

    def gen_wfefield_coeff(self, force=False, save=True, **kwargs):
        return _gen_wfefield_coeff(self, force=force, save=save, **kwargs)
        """ Fit WFE field-dependent coefficients
        """

    def calc_psf_from_coeff(self, sp=None, return_oversample=False, 
        wfe_drift=None, coord_vals=None, coord_frame='tel', **kwargs):
        """ Create PSF image from coefficients
        
        Create a PSF image from instrument settings. The image is noiseless and
        doesn't take into account any non-linearity or saturation effects, but is
        convolved with the instrument throughput. Pixel values are in counts/sec.
        The result is effectively an idealized slope image (no background).

        Returns a single image or list of images if sp is a list of spectra. 
        By default, it returns only the detector-sampled PSF, but setting 
        return_oversample=True will also return a set of oversampled images
         as a second output.

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
            If True, then also returns the oversampled version of the PSF
        use_bg_psf : bool
            If a coronagraphic observation, off-center PSF is different.
        wfe_drift : float or None
            Wavefront error drift amplitude in nm.
        coord_vals : tuple or None
            Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
        coord_frame : str
            Type of input coordinates. 

                * 'tel': arcsecs V2,V3
                * 'sci': pixels, in conventional DMS axes orientation
                * 'det': pixels, in raw detector read out axes orientation
                * 'idl': arcsecs relative to aperture reference location.

        return_hdul : bool
            TODO: Return PSFs in an HDUList rather than set of arrays
        """        

        return _calc_psf_from_coeff(self, sp=sp, return_oversample=return_oversample, 
                                    wfe_drift=wfe_drift, coord_vals=coord_vals, 
                                    coord_frame=coord_frame, **kwargs)


# MIRI Subclass
class MIRI_ext(webbpsf_MIRI):
    
    def __init__(self, filter=None, pupil_mask=None, image_mask=None, 
                 fov_pix=None, oversample=None, auto_gen_coeffs=False):
        
        webbpsf_MIRI.__init__(self)
        _init_inst(self, filter=filter, pupil_mask=pupil_mask, image_mask=image_mask,
                   fov_pix=fov_pix, oversample=oversample)

        if auto_gen_coeffs:
            self.gen_psf_coeff()
            self.gen_wfedrift_coeff()
            self.gen_wfefield_coeff()

    @property
    def save_dir(self):
        """Coefficient save directory"""
        # Default name
        if self._save_dir is None:
            # Name to save array of oversampled coefficients
            inst_name = self.name
            save_dir = conf.WEBBPSF_EXT_PATH + f'{inst_name}/psf_coeffs/'
            # Create directory if it doesn't already exist
            if not os.path.isdir(save_dir):
               os.makedirs(save_dir)
        else:
            save_dir = self._save_dir
        return save_dir
    @save_dir.setter
    def save_dir(self, value):
        self._save_dir = value
        
    @property
    def save_name(self):
        """Coefficient file name"""
        save_name = self._save_name
        save_name = self.gen_save_name() if save_name is None else save_name
        return save_name
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
    def npsf(self):
        """Number of wavelengths/PSFs to fit"""
        npsf = self._npsf
        
        # Default to 10 PSF simulations per um
        w1 = self.bandpass.wave.min() / 1e4
        w2 = self.bandpass.wave.max() / 1e4
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
        """Number of wavelengths/PSFs to fit"""
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
        return True if self._quick is None else self._quick
    @quick.setter
    def quick(self, value):
        """Perform quicker coeff calculation over limited bandwidth?"""
        _check_list(value, [True, False], 'quick')
        self._quick = value

    @webbpsf_MIRI.detector_position.setter
    def detector_position(self, position):
        try:
            x, y = map(int, position)
        except ValueError:
            raise ValueError("Detector pixel coordinates must be a pair of numbers, not {}".format(position))
        self._detector_position = (int(position[0]), int(position[1]))
    
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

    def _addAdditionalOptics(self, optsys, oversample=2):
        """Add coronagraphic or spectrographic optics for MIRI.
        Semi-analytic coronagraphy algorithm used for the Lyot only.

        """

        # For MIRI coronagraphy, all the coronagraphic optics are rotated the same
        # angle as the instrument is, relative to the primary. So they see the unrotated
        # telescope pupil. Likewise the LRS grism is rotated but its pupil stop is not.
        #
        # We model this by just not rotating till after the coronagraph. Thus we need to
        # un-rotate the primary that was already created in get_optical_system.
        # This approach is required computationally so we can work in an unrotated frame
        # aligned with the FQPM axes.

        defaultpupil = optsys.planes.pop(2)  # throw away the rotation of the entrance pupil we just added

        if self.include_si_wfe:
            # temporarily remove the SI internal aberrations
            # from the system - will add back in after the
            # coronagraph planes.
            miri_aberrations = optsys.planes.pop(2)

        # Add image plane mask
        # For the MIRI FQPMs, we require the star to be centered not on the middle pixel, but
        # on the cross-hairs between four pixels. (Since that is where the FQPM itself is centered)
        # This is with respect to the intermediate calculation pixel scale, of course, not the
        # final detector pixel scale.
        if ((self.image_mask is not None) and ('FQPM' in self.image_mask)) or ('force_fqpm_shift' in self.options):
            optsys.add_pupil(poppy.FQPM_FFT_aligner())

        # Allow arbitrary offsets of the focal plane masks with respect to the pixel grid origin;
        # In most use cases it's better to offset the star away from the mask instead, using
        # options['source_offset_*'], but doing it this way instead is helpful when generating
        # the Pandeia ETC reference PSF library.
        offsets = {'shift_x': self.options.get('mask_shift_x', None),
                   'shift_y': self.options.get('mask_shift_y', None)}

        def make_fqpm_wrapper(name, wavelength):
            container = poppy.CompoundAnalyticOptic(name=name,
                                                    opticslist=[poppy.IdealFQPM(wavelength=wavelength, name=self.image_mask, **offsets),
                                                                poppy.SquareFieldStop(size=24, rotation=self._rotation, **offsets)])
            return container

        if self.image_mask == 'FQPM1065':
            optsys.add_image(make_fqpm_wrapper("MIRI FQPM 1065", 10.65e-6))
            trySAM = False
        elif self.image_mask == 'FQPM1140':
            optsys.add_image(make_fqpm_wrapper("MIRI FQPM 1140", 11.40e-6))
            trySAM = False
        elif self.image_mask == 'FQPM1550':
            optsys.add_image(make_fqpm_wrapper("MIRI FQPM 1550", 15.50e-6))
            trySAM = False
        elif self.image_mask == 'LYOT2300':
            # diameter is 4.25 (measured) 4.32 (spec) supposedly 6 lambda/D
            # optsys.add_image(function='CircularOcculter',radius =4.25/2, name=self.image_mask)
            # Add bar occulter: width = 0.722 arcsec (or perhaps 0.74, Dean says there is ambiguity)
            # optsys.add_image(function='BarOcculter', width=0.722, angle=(360-4.76))
            # position angle of strut mask is 355.5 degrees  (no = =360 -2.76 degrees
            # optsys.add_image(function='fieldstop',size=30)
            container = poppy.CompoundAnalyticOptic(name="MIRI Lyot Occulter",
                                            opticslist=[poppy.CircularOcculter(radius=4.25 / 2, name=self.image_mask, **offsets),
                                                        poppy.BarOcculter(width=0.722, **offsets),
                                                        poppy.SquareFieldStop(size=30, rotation=self._rotation, **offsets)])
            optsys.add_image(container)
            trySAM = False  # FIXME was True - see https://github.com/mperrin/poppy/issues/169
            SAM_box_size = [5, 20]
        elif self.image_mask == 'LRS slit':
            # one slit, 5.5 x 0.6 arcsec in height (nominal)
            #           4.7 x 0.51 arcsec (measured for flight model. See MIRI-TR-00001-CEA)
            #
            # Per Klaus Pontoppidan: The LRS slit is aligned with the detector x-axis, so that the
            # dispersion direction is along the y-axis.
            optsys.add_image(optic=poppy.RectangularFieldStop(width=4.7, height=0.51,
                                                              rotation=self._rotation, name=self.image_mask, **offsets))
            trySAM = False
        else:
            optsys.add_image()
            trySAM = False

        if ((self.image_mask is not None and 'FQPM' in self.image_mask)
                or 'force_fqpm_shift' in self.options):
            optsys.add_pupil(poppy.FQPM_FFT_aligner(direction='backward'))

        # add pupil plane mask
        shift_x, shift_y = self._get_pupil_shift()
        rotation = self.options.get('pupil_rotation', None)

        if self.pupil_mask == 'MASKFQPM':
            optsys.add_pupil(transmission=self._datapath + "/optics/MIRI_FQPMLyotStop.fits.gz",
                             name=self.pupil_mask,
                             flip_y=True, shift_x=shift_x, shift_y=shift_y, rotation=rotation)
            optsys.planes[-1].wavefront_display_hint = 'intensity'
        elif self.pupil_mask == 'MASKLYOT':
            optsys.add_pupil(transmission=self._datapath + "/optics/MIRI_LyotLyotStop.fits.gz",
                             name=self.pupil_mask,
                             flip_y=True, shift_x=shift_x, shift_y=shift_y, rotation=rotation)
            optsys.planes[-1].wavefront_display_hint = 'intensity'
        elif self.pupil_mask == 'P750L LRS grating' or self.pupil_mask == 'P750L':
            optsys.add_pupil(transmission=self._datapath + "/optics/MIRI_LRS_Pupil_Stop.fits.gz",
                             name=self.pupil_mask,
                             flip_y=True, shift_x=shift_x, shift_y=shift_y, rotation=rotation)
            optsys.planes[-1].wavefront_display_hint = 'intensity'
        else:  # all the MIRI filters have a tricontagon outline, even the non-coron ones.
            optsys.add_pupil(transmission=self._WebbPSF_basepath + "/tricontagon.fits.gz",
                             name='filter cold stop', shift_x=shift_x, shift_y=shift_y, rotation=rotation)
            # FIXME this is probably slightly oversized? Needs to have updated specifications here.

        if self.include_si_wfe:
            # now put back in the aberrations we grabbed above.
            optsys.add_pupil(miri_aberrations)

        optsys.add_rotation(self._rotation, hide=True)
        optsys.planes[-1].wavefront_display_hint = 'intensity'

        return (optsys, trySAM, SAM_box_size if trySAM else None)

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
        offsets = {'shift_x': self.options.get('mask_shift_x', None),
                   'shift_y': self.options.get('mask_shift_y', None)}
        
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
            im = mask.get_transmission(wave)
        elif self.image_mask == 'LRS slit':
            full_pad = 2*np.max(np.abs(xy_rot(2.5, 2.5, rot2)))
            npix = int(full_pad / pixelscale + 0.5) if npix is None else npix
            wave = poppy.Wavefront(wavelength=23e-6, npix=npix, pixelscale=pixelscale)
            mask = poppy.RectangularFieldStop(width=4.7, height=0.51, rotation=rot2, 
                                              name=self.image_mask, **offsets)
            im = mask.get_transmission(wave)
        else:
            im = np.ones([npix,npix])
        
        return im
        
    def get_opd_info(self, opd=None, HDUL_to_OTELM=True):
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
        return _get_opd_info(self, opd=opd, HDUL_to_OTELM=HDUL_to_OTELM)
    
    def drift_opd(self, wfe_drift, opd=None):
        """
        A quick method to drift the pupil OPD. This function applies 
        some WFE drift to input OPD file by breaking up the wfe_drift 
        attribute into thermal, frill, and IEC components. If we want 
        more realistic time evolution, then we should use the procedure 
        in dev_utils/WebbPSF_OTE_LM.ipynb to create a time series of OPD
        maps, which can then be passed directly to create unique PSFs.
        
        This outputs an OTE Linear Model. 
        In order to update instrument class:
            >>> opd_new = inst.drift_opd()
            >>> inst.pupilopd = opd_new
            >>> inst.pupil = opd_new
        """
        return _drift_opd(self, wfe_drift, opd=opd)

    def gen_save_name(self, wfe_drift=0):
        """
        Generate save name for polynomial coefficient output file.
        """
        return _gen_save_name(self, wfe_drift=wfe_drift)

    def gen_psf_coeff(self, wfe_drift=0, nproc=None, force=False, 
                      save=True, return_results=False, return_extras=False):
        """Generate PSF coefficients

        Creates a set of coefficients that will generate a simulated PSF at any
        arbitrary wavelength. This function first uses simulates a number of
        evenly spaced PSFs throughout some specified bandpass. An nth-degree 
        polynomial is then fit to each oversampled pixel using a linear-least 
        squares fitting routine. The final set of coefficients for each pixel 
        is returned as an image cube. The returned set of coefficient can be 
        used to produce a set of PSFs by:

        >>> psfs = jl_poly(waves, coeffs)

        where 'waves' can be a scalar, nparray, list, or tuple. All wavelengths
        are in microns.

        Parameters
        ----------
        wfe_drift : float
            Wavefront error drift amplitude in nm.
        force : bool
            Forces a recalculation of PSF even if saved PSF exists. (default: False)
        save : bool
            Save the resulting PSF coefficients to a file? (default: True)
        """
        
        return _gen_psf_coeff(self, nproc=nproc, wfe_drift=wfe_drift, force=force, save=save, 
                              return_results=return_results, return_extras=return_extras)

    def gen_wfedrift_coeff(self, force=False, save=True, **kwargs):
        """ Fit WFE drift coefficients

        This function finds a relationship between PSF coefficients in the presence of WFE drift. 
        For a series of WFE drift values, we generate corresponding PSF coefficients and fit a 
        polynomial relationship to the residual values. This allows us to quickly modify a nominal 
        set of PSF image coefficients to generate a new PSF where the WFE has drifted by some amplitude.
        
        It's Legendre's all the way down...

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
        return _gen_wfedrift_coeff(self, force=force, save=save, **kwargs)

    def gen_wfefield_coeff(self, force=False, save=True, **kwargs):
        return _gen_wfefield_coeff(self, force=force, save=save, **kwargs)

    def calc_psf_from_coeff(self, sp=None, return_oversample=False, 
        wfe_drift=None, coord_vals=None, coord_frame='tel', **kwargs):
        """ Create PSF image from coefficients
        
        Create a PSF image from instrument settings. The image is noiseless and
        doesn't take into account any non-linearity or saturation effects, but is
        convolved with the instrument throughput. Pixel values are in counts/sec.
        The result is effectively an idealized slope image (no background).

        Returns a single image or list of images if sp is a list of spectra. 
        By default, it returns only the detector-sampled PSF, but setting 
        return_oversample=True will also return a set of oversampled images
         as a second output.

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
            If True, then also returns the oversampled version of the PSF
        use_bg_psf : bool
            If a coronagraphic observation, off-center PSF is different.
        wfe_drift : float or None
            Wavefront error drift amplitude in nm.
        coord_vals : tuple or None
            Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
        coord_frame : str
            Type of input coordinates. 

                * 'tel': arcsecs V2,V3
                * 'sci': pixels, in conventional DMS axes orientation
                * 'det': pixels, in raw detector read out axes orientation
                * 'idl': arcsecs relative to aperture reference location.

        return_hdul : bool
            TODO: Return PSFs in an HDUList rather than set of arrays
        """        

        return _calc_psf_from_coeff(self, sp=sp, return_oversample=return_oversample, 
                                    wfe_drift=wfe_drift, coord_vals=coord_vals, 
                                    coord_frame=coord_frame, **kwargs)


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

def _check_fitsgz(self, opd_file):
    """
    WebbPSF FITS files can be either .fits or compressed .gz. 
    Search for .fits by default, then .fits.gz.
    """
    inst_str = self.name

    # .fits or .fits.gz?
    opd_dir = os.path.join(self._datapath,'OPD')
    opd_fullpath = os.path.join(opd_dir, opd_file)

    # Check if file exists 
    if not os.path.exists(opd_fullpath):
        opd_file_alt = opd_file + '.gz'
        opd_path_alt = os.path.join(opd_dir, opd_file_alt)
        if not os.path.exists(opd_path_alt):
            err_msg = f'Cannot find either {opd_file} or {opd_file_alt} in {opd_dir}'
            raise OSError(err_msg)
        else:
            opd_file = opd_file_alt

    return opd_file

def _init_inst(self, filter=None, pupil_mask=None, image_mask=None, 
               fov_pix=None, oversample=None):
    """
    Setup for specific instrument during init state
    """

    # Add grisms as pupil options
    if self.name=='NIRCam':
        self.pupil_mask_list = self.pupil_mask_list + ['GRISM0', 'GRISM90', 'GRISMC', 'GRISMR']
    elif self.name=='NIRISS':
        self.pupil_mask_list = self.pupil_mask_list + ['GR150C', 'GR150R']
    self.pupil_mask_list

    if filter is not None:
        self.filter = filter
    if pupil_mask is not None:
        self.pupil_mask = pupil_mask
    if image_mask is not None:
        self.image_mask = image_mask
        
    # Don't include SI WFE error for coronagraphy
    if self.name=='MIRI':
        self.include_si_wfe = False if self.is_coron else True
    else:
        self.include_si_wfe = True
    
    # Settings for fov_pix and oversample
    # Default odd for normal imaging, even for coronagraphy
    # TODO: Do these even/odd settings make sense?
    if fov_pix is None:
        fov_pix = 128 if self.is_coron else 129
    self._fov_pix = fov_pix
    self._oversample = oversample
    
    # Setting these to one choose default values at runtime
    self._npsf = None
    self._ndeg = None
    
    # Legendre polynomials are more stable
    self.use_legendre = True
    
    # Turning on quick perform fits over filter bandpasses independently
    # The smaller wavelength range requires fewer monochromaic wavelengths
    # and lower order polynomial fits
    self._quick = None
    
    # Set up initial OPD file info
    opd_name = f'OPD_RevW_ote_for_{self.name}_predicted.fits'
    opd_name = _check_fitsgz(self, opd_name)
    self._opd_default = (opd_name, 0)
    self.pupilopd = self._opd_default
    
    # Name to save array of oversampled coefficients
    self._save_dir = None
    self._save_name = None
    
    # No jitter for coronagraphy
    self.options['jitter'] = None if self.is_coron else 'gaussian'
    self.options['jitter_sigma'] = 0.003

    # Max FoVs for calculating drift and field-dependent coefficient residuals
    # Any pixels beyond this size will be considered to have 0 residual difference
    self._fovmax_wfedrift = 256
    self._fovmax_wfefield = 128

    self.psf_coeff = None
    self.psf_coeff_header = None
    self._psf_coeff_mod = {
        'wfe_drift': None, 'wfe_drift_lxmap': None,
        'si_field': None, 'si_field_v2grid': None, 'si_field_v3grid': None,
    } 

def _gen_save_name(self, wfe_drift=0):
    """
    Create save name for polynomial coefficients output file.
    """
    
    # Prepend filter name if using quick keyword
    fstr = '{}_'.format(self.filter) if self.quick else ''
    # Mask and pupil names
    mstr = 'NONE' if self.image_mask is None else self.image_mask
    pstr = 'CLEAR' if self.pupil_mask is None else self.pupil_mask
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
        bar_offset = self.options.get('bar_offset', None)
        bar_str = '' if bar_offset is None else '_bar{:.1f}'.format(bar_offset)
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
    offset_theta = self.options.get('offset_theta', 0)
    if offset_r is None: 
        offset_r = 0
    if offset_theta is None: 
        offset_theta = 0
    rth_str = f'r{offset_r:.2f}_th{offset_theta:+.1f}'
    
    # Mask offsetting
    coron_offset_x = self.options.get('mask_shift_x', 0)
    coron_offset_y = self.options.get('mask_shift_y', 0)
    if coron_offset_x is None: 
        coron_offset_x = 0
    if coron_offset_y is None: 
        coron_offset_y = 0
    moff_str1 = '' if coron_offset_x==0 else f'_mx{coron_offset_x:.3f}'
    moff_str2 = '' if coron_offset_y==0 else f'_my{coron_offset_y:.3f}'
    moff_str = moff_str1 + moff_str2
    
    opd_dict = self.get_opd_info()
    opd_str = opd_dict['opd_str']

    if wfe_drift>0:
        opd_str = '{}-{:.0f}nm'.format(opd_str,wfe_drift)
    
    fname = f'{fmp_str}_pix{fov_pix}_os{osamp}_jsig{jsig_mas:.0f}_{rth_str}{moff_str}{bar_str}_{opd_str}'
    
    # Add SI WFE tag if included
    if self.include_si_wfe:
        fname = fname + '_siwfe'

    if self.use_legendre:
        fname = fname + '_legendre'

    fname = fname + '.fits'
    
    return fname


def _get_opd_info(self, opd=None, HDUL_to_OTELM=True):
    """
    Parse out OPD information for a given OPD, which can be a 
    file name, tuple (file,slice), HDUList, or OTE Linear Model. 
    Returns dictionary of some relevant information for logging purposes.
    The dictionary has an OPD version as an OTE LM.
    
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
        
    # If OPD is None or a string, make into tuple
    if opd is None:  # Default OPD
        opd = self._opd_default
    elif isinstance(opd, six.string_types):
        opd = (opd, 0)

    # Change log levels to WARNING for pyNRC, WebbPSF, and POPPY
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
        opd_name = 'OPD from OTE LM'
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
        opd = OTE_Linear_Model_WSS(name=name, opd=hdul, opd_index=opd_num, transmission=self.pupil)
        
    setup_logging(log_prev, verbose=False)

    out_dict = {'opd_name':opd_name, 'opd_num':opd_num, 'opd_str':opd_str, 'pupilopd':opd}
    return out_dict

def _drift_opd(self, wfe_drift, opd=None):
    """
    A quick method to drift the pupil OPD. This function applies 
    some WFE drift to input OPD file by breaking up the wfe_drift 
    attribute into thermal, frill, and IEC components. If we want 
    more realistic time evolution, then we should use the procedure 
    in dev_utils/WebbPSF_OTE_LM.ipynb to create a time series of OPD
    maps, which can then be passed directly to create unique PSFs.
    
    This outputs an OTE Linear Model. 
    In order to update instrument class:
        >>> opd_new = inst.drift_opd()
        >>> inst.pupilopd = opd_new
        >>> inst.pupil = opd_new
    """
    
    # Get Pupil OPD info and convert to OTE LM
    opd_dict = self.get_opd_info(opd)
    opd_name = opd_dict['opd_name']
    opd_num  = opd_dict['opd_num']
    opd_str  = opd_dict['opd_str']
    opd      = opd_dict['pupilopd']
        
    # If there is wfe_drift, create a OTE Linear Model
    wfe_dict = {'therm':0, 'frill':0, 'iec':0, 'opd':opd}
    if (wfe_drift > 0):
        _log.info('Performing WFE drift of {}nm'.format(wfe_drift))

        # Apply WFE drift to OTE Linear Model (Amplitude of frill drift)
        # self.pupilopd = opd
        # self.pupil = opd

        # Split WFE drift amplitude between three processes
        # 1) IEC Heaters; 2) Frill tensioning; 3) OTE Thermal perturbations
        # Give IEC heaters 1 nm 
        wfe_iec = 1 if np.abs(wfe_drift) > 2 else 0

        # Split remainder evenly between frill and OTE thermal slew
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
        opd.apply_iec_drift(wfe_iec, delay_update=True)
        # Apply frill
        opd.apply_frill_drift(wfe_frill, delay_update=True)

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

    return wfe_dict


def _gen_psf_coeff(self, nproc=None, wfe_drift=0, force=False, save=True, 
                   return_results=False, return_extras=False):
    
    save_name = self.save_name
    outfile = self.save_dir + save_name
    # Load data from already saved FITS file
    if os.path.exists(outfile) and (not force):
        if return_extras:
            _log.warn("return_extras only valid if coefficient files does not exist or force=True")

        hdul = fits.open(outfile)
        data = hdul[0].data.astype(np.float)
        hdr  = hdul[0].header
        hdul.close()

        # Output if return_results=True, otherwise save to attributes
        if return_results:
            return data, hdr
        else:
            self.psf_coeff = data
            self.psf_coeff_header = hdr
            return
    
    temp_str = 'and saving' if save else 'but not saving'
    _log.info(f'Generating {temp_str} PSF coefficient')

    # Change log levels to WARNING for pyNRC, WebbPSF, and POPPY
    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    
    w1 = self.bandpass.wave.min() / 1e4
    w2 = self.bandpass.wave.max() / 1e4
    npsf = self.npsf
    waves = np.linspace(w1, w2, npsf)
        
    fov_pix = self.fov_pix #if fov_pix is None else fov_pix
    oversample = self.oversample #if oversample is None else oversample
            
    # Get OPD info and convert to OTE LM
    opd_dict = self.get_opd_info(HDUL_to_OTELM=True)
    opd_name = opd_dict['opd_name']
    opd_num  = opd_dict['opd_num']
    opd_str  = opd_dict['opd_str']
    opd      = opd_dict['pupilopd']
    
    # Drift OPD
    if wfe_drift>0:
        wfe_dict = self.drift_opd(wfe_drift, opd=opd)
    else:
        wfe_dict = {'therm':0, 'frill':0, 'iec':0, 'opd':opd}
    opd_new = wfe_dict['opd']
    # Save copies
    pupilopd_orig = deepcopy(self.pupilopd)
    pupil_orig = deepcopy(self.pupil)
    self.pupilopd = opd_new
    self.pupil = opd_new
    
    # How many processors to split into?
    if nproc is None:
        nproc = nproc_use(fov_pix, oversample, npsf)
    _log.debug('nprocessors: {}; npsf: {}'.format(nproc, npsf))

    setup_logging('WARN', verbose=False)
    t0 = time.time()
    # Setup the multiprocessing pool and arguments to pass to each pool
    worker_arguments = [(self, wlen, fov_pix, oversample) for wlen in waves]
    if nproc > 1:
        pool = mp.Pool(nproc)
        # Pass arguments to the helper function

        try:
            hdu_arr = pool.map(_wrap_coeff_for_mp, worker_arguments)
            if hdu_arr[0] is None:
                raise RuntimeError('Returned None values. Issue with multiprocess or WebbPSF??')

        except Exception as e:
            _log.error('Caught an exception during multiprocess.')
            _log.error('Closing multiprocess pool.')
            pool.terminate()
            pool.close()
            raise e

        else:
            _log.debug('Closing multiprocess pool.')
            pool.close()
    else:
        # Pass arguments to the helper function
        hdu_arr = []
        for wa in worker_arguments:
            hdu = _wrap_coeff_for_mp(wa)
            if hdu is None:
                raise RuntimeError('Returned None values. Issue with WebbPSF??')
            hdu_arr.append(hdu)
    t1 = time.time()
    
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

    # Turn results into an numpy array (npsf,ny,nx)
    images = np.array(images)

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
        'EXTNAME', 'OVERSAMP', 'DET_SAMP', 'PIXELSCL', 'FOV',     
        'INSTRUME', 'FILTER', 'PUPIL', 'CORONMSK',
        'WAVELEN', 'DIFFLMT', 'APERNAME', 'MODULE', 'CHANNEL', 'PILIN',
        'DET_NAME', 'DET_X', 'DET_Y', 'DET_V2', 'DET_V3',  
        'GRATNG14', 'GRATNG23', 'FLATTYPE', 'CCCSTATE', 'TACQNAME',
        'PUPILINT', 'PUPILOPD', 'OPD_FILE', 'OPDSLICE', 'TEL_WFE', 
        'SI_WFE', 'SIWFETYP', 'SIWFEFPT',
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
    offset_theta = self.options.get('offset_theta', 'None')
    
    # Mask offsetting
    coron_offset_x = self.options.get('mask_shift_x', 'None')
    coron_offset_y = self.options.get('mask_shift_y', 'None')
    bar_offset = self.options.get('bar_offset', 'None')
        
    # Jitter settings
    jitter = self.options.get('jitter')
    jitter_sigma = self.options.get('jitter_sigma', 0)
            
    # gen_psf_coeff() Keyword Values
    hdr['FOVPIX'] = (fov_pix, 'WebbPSF pixel FoV')
    hdr['OSAMP']  = (oversample, 'WebbPSF pixel oversample')
    hdr['NPSF']   = (npsf, 'Number of wavelengths to calc')
    hdr['NDEG']   = (ndeg, 'Polynomial fit degree')
    hdr['WAVE1']  = (w1, 'First wavelength in calc')
    hdr['WAVE2']  = (w2, 'Last of wavelength in calc')
    hdr['LEGNDR'] = (use_legendre, 'Legendre polynomial fit?')
    hdr['OFFR']  = (offset_r, 'Radial offset')
    hdr['OFFTH'] = (offset_theta, 'Position angle OFFR (CCW)')
    hdr['MASKOFFX'] = (coron_offset_x, 'Image mask shift in x (arcsec)')
    hdr['MASKOFFY'] = (coron_offset_y, 'Image mask shift in y (arcsec)')
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
        from astropy.utils.exceptions import AstropyWarning
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            hdu.writeto(outfile, overwrite=True)

    if return_results==False:
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

def _gen_wfedrift_coeff(self, force=False, save=True, return_results=False,
                        wfe_list=[0,1,2,5,10,20,40], **kwargs):

    # fov_pix should not be more than some size to preserve memory
    fov_max = self._fovmax_wfedrift if self.oversample<=4 else self._fovmax_wfedrift / 2 
    fov_pix_orig = self.fov_pix
    if self.fov_pix>fov_max:
        self.fov_pix = fov_max if (self.fov_pix % 2 == 0) else fov_max + 1

    # Name to save array of oversampled coefficients
    save_dir = self.save_dir
    save_name = os.path.splitext(self.save_name)[0] + '_wfedrift.npz'
    outname = save_dir + save_name

    # Load file if it already exists
    if (not force) and os.path.exists(outname):
        out = np.load(outname)
        if return_results:
            return out['arr_0'], out['arr_1']
        else:
            self._psf_coeff_mod['wfe_drift'] = out['arr_0']
            self._psf_coeff_mod['wfe_drift_lxmap'] = out['arr_1']
            return

    _log.warn('Generating WFE Drift coefficients. This may take some time...')

    # Cycle through WFE drifts for fitting
    wfe_list = np.array(wfe_list)
    npos = len(wfe_list)

    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)
    # Calculate residuals
    cf_wfe = []
    for wfe_drift in tqdm(wfe_list, leave=False):
        cf, _ = self.gen_psf_coeff(wfe_drift=wfe_drift, force=True, save=False, return_results=True)
        cf_wfe.append(cf)
    cf_wfe = np.array(cf_wfe) - cf_wfe[0]
    setup_logging(log_prev, verbose=False)

    # Fit each pixel with a polynomial and save the coefficient
    cf_shape = cf_wfe.shape[1:]
    cf_wfe = cf_wfe.reshape([npos, -1])
    lxmap = np.array([np.min(wfe_list), np.max(wfe_list)])
    cf_fit = jl_poly_fit(wfe_list, cf_wfe, deg=4, use_legendre=True, lxmap=lxmap)
    cf_fit = cf_fit.reshape([-1, cf_shape[0], cf_shape[1], cf_shape[2]])

    if save:
        np.savez(outname, cf_fit, lxmap)
    _log.info('Done.')

    # Return fov_pix to original size
    self.fov_pix = fov_pix_orig

    # Options to return results from function
    if return_results:
        return cf_fit, lxmap
    else:
        self._psf_coeff_mod['wfe_drift'] = cf_fit
        self._psf_coeff_mod['wfe_drift_lxmap'] = lxmap


def _gen_wfefield_coeff(self, force=False, save=True, return_results=False, return_raw=False, **kwargs):

    if (self.include_si_wfe==False) or (self.is_coron):
        _log.warn("Skipping WFE field dependence...")
        if self.include_si_wfe==False:
            _log.warn("   `include_si_wfe` attribute is set to False")
        if self.is_coron:
            _log.warn("   Coronagraphic image mask in place")
        return

    # fov_pix should not be more than some size to preserve memory
    fov_max = self._fovmax_wfefield if self.oversample<=4 else self._fovmax_wfefield / 2 
    fov_pix_orig = self.fov_pix
    if self.fov_pix>fov_max:
        self.fov_pix = fov_max if (self.fov_pix % 2 == 0) else fov_max + 1

    # Name to save array of oversampled coefficients
    save_dir = self.save_dir
    save_name = os.path.splitext(self.save_name)[0] + '_wfefields.npz'
    outname = save_dir + save_name

    # Load file if it already exists
    if (not force) and os.path.exists(outname):
        out = np.load(outname)
        if return_results:
            return out['arr_0'], out['arr_1'], out['arr_2']
        else:
            self._psf_coeff_mod['si_field'] = out['arr_0']
            self._psf_coeff_mod['si_field_v2grid'] = out['arr_1']
            self._psf_coeff_mod['si_field_v3grid'] = out['arr_2']
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
    else: 
        if self.name=='MIRI':
            apname = 'MIRIM_FULL'
        else:
            raise NotImplementedError("Field Variations not implemented for {}".format(self.name))

        ap = self.siaf[apname]
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
    ap = self.siaf[self.aperturename]
    xsci_all, ysci_all = ap.convert(v2_all*60, v3_all*60, 'tel', 'sci')

    log_prev = conf.logging_level
    setup_logging('WARN', verbose=False)

    # Initial settings
    coeff0 = self.psf_coeff
    x0, y0 = self.detector_position

    # Calculate new coefficients at each position
    cf_fields = []
    for xsci, ysci in tqdm(zip(xsci_all, ysci_all), total=npos):
        # Update saved detector position and calculate PSF coeff
        self.detector_position = (xsci, ysci)
        cf, _ = self.gen_psf_coeff(force=True, save=False, return_results=True)
        cf_fields.append(cf)

    # Reset to initial values
    self.detector_position = (x0,y0)
    setup_logging(log_prev, verbose=False)

    # Get residuals
    cf_fields_resid = np.array(cf_fields) - coeff0

    if return_raw:
        return cf_fields_resid, v2_all, v3_all

    # Create an evenly spaced grid of V2/V3 coordinates
    nv23 = 8
    v2grid = np.linspace(v2_min, v2_max, num=nv23)
    v3grid = np.linspace(v3_min, v3_max, num=nv23)

    # Interpolate onto an evenly space grid
    res = make_coeff_resid_grid(v2_all, v3_all, cf_fields_resid, v2grid, v3grid)
    if save: 
        np.savez(outname, *res)

    if return_results:
        return res
    else:
        self._psf_coeff_mod['si_field'] = res[0]
        self._psf_coeff_mod['si_field_v2grid'] = res[1]
        self._psf_coeff_mod['si_field_v3grid'] = res[2]



def _calc_psf_from_coeff(self, sp=None, return_oversample=False, 
    wfe_drift=None, coord_vals=None, coord_frame='tel', return_hdul=True, **kwargs):
    """PSF Image from polynomial coefficients
    
    Create a PSF image from instrument settings. The image is noiseless and
    doesn't take into account any non-linearity or saturation effects, but is
    convolved with the instrument throughput. Pixel values are in counts/sec.
    The result is effectively an idealized slope image (no background).

    If no spectral dispersers (grisms or DHS), then this returns a single
    image or list of images if sp is a list of spectra. By default, it returns
    only the detector-sampled PSF, but setting return_oversample=True will
    also return a set of oversampled images as a second output.

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
        If True, then also returns the oversampled version of the PSF
    use_bg_psf : bool
        If a coronagraphic observation, off-center PSF is different.
    wfe_drift : float or None
        Wavefront error drift amplitude in nm.
    coord_vals : tuple or None
        Coordinates (in arcsec or pixels) to calculate field-dependent PSF.
    coord_frame : str
        Type of input coordinates. 

            * 'tel': arcsecs V2,V3
            * 'sci': pixels, in conventional DMS axes orientation
            * 'det': pixels, in raw detector read out axes orientation
            * 'idl': arcsecs relative to aperture reference location.

    return_hdul : bool
        TODO: Return PSFs in an HDUList rather than set of arrays
    """        

    psf_coeff_hdr = self.psf_coeff_header
    psf_coeff     = self.psf_coeff

    if psf_coeff is None:
        _log.warning("You must first run `gen_psf_coeff` calculating PSFs.")
        return

    # Spectrographic Mode?
    is_spec = False
    if (self.name=='NIRCam') or (self.name=='NIRISS'):
        is_spec = True if self.is_grism else False
    elif (self.name=='MIRI') or (self.name=='NIRSpec'):
        is_spec = True if self.is_slitspec else False

    # Coeff modification variable
    psf_coeff_mod = 0 

    if wfe_drift is None: 
        wfe_drift = 0
    
    # Modify PSF coefficients based on WFE drift
    psf_coeff_mod += _coeff_mod_wfe_drift(self, wfe_drift)

    # Modify PSF coefficients based on field-dependence
    cf_mod, nfield = _coeff_mod_wfe_field(self, coord_vals, coord_frame)
    psf_coeff_mod += cf_mod

    # Add modifications to coefficients
    psf_coeff = psf_coeff + psf_coeff_mod
    del psf_coeff_mod

    # if multiple field points were present, we want to return PSF for each location
    if nfield>1:
        psf_all = []
        for ii in trange(nfield, leave=False):
            # Just a single spectrum? Or unique spectrum at each field point?
            sp_norm = sp[ii] if((sp is not None) and (len(sp)==nfield)) else sp
            res = gen_image_from_coeff(self, psf_coeff[ii], psf_coeff_hdr, sp_norm=sp_norm,
                                       return_oversample=return_oversample)
            # For grisms (etc), the wavelength solution is the same for each field point
            wave, psf = res if is_spec else (None, res)
            psf_all.append(psf)

        if return_hdul:
            xvals, yvals = coord_vals
            hdul = fits.HDUList()
            for ii, psf in enumerate(psf_all):
                hdr = psf_coeff_hdr.copy()
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
            output = (wave, psf_all) if is_spec else psf_all
    else:
        res = gen_image_from_coeff(self, psf_coeff, psf_coeff_hdr, sp_norm=sp,
                                   return_oversample=return_oversample)

        if return_hdul:
            # For grisms (etc), the wavelength solution is the same for each field point
            wave, psf = res if is_spec else (None, res)

            hdr = psf_coeff_hdr.copy()
            hdr['WFEDRIFT'] = (wfe_drift, '[nm] WFE drift amplitude')
            if coord_vals is not None:
                cunits = 'pixels' if ('sci' in coord_frame) or ('det' in coord_frame) else 'arcsec'
                hdr['XVAL']   = (coord_vals[0], f'[{cunits}] Input X coordinate')
                hdr['YVAL']   = (coord_vals[0], f'[{cunits}] Input Y coordinate')
                hdr['CFRAME'] = (coord_frame, 'Specified coordinate frame')
            hdul = fits.HDUList([fits.PrimaryHDU(data=psf, header=hdr)])
            # Append wavelength solution
            if wave is not None:
                hdul.append(fits.ImageHDU(data=wave, name='Wavelengths'))
            output = hdul
        else:
            output = res
    
    return output

def _coeff_mod_wfe_drift(self, wfe_drift):
    """
    Modify PSF polynomial coefficients as a function of WFE drift.
    """

    # Modify PSF coefficients based on WFE drift
    if wfe_drift==0:
        cf_mod = 0 # Don't modify coefficients
    elif (self._psf_coeff_mod['wfe_drift'] is None):
        _log.warning("You must run `gen_wfedrift_coeff` first before setting the wfe_drift parameter.")
        _log.warning("Will continue assuming `wfe_drift=0`.")
        cf_mod = 0
    else:

        psf_coeff_hdr = self.psf_coeff_header
        psf_coeff     = self.psf_coeff

        cf_fit = self._psf_coeff_mod['wfe_drift'] 
        lxmap  = self._psf_coeff_mod['wfe_drift_lxmap'] 

        # Fit function
        cf_fit_shape = cf_fit.shape
        cf_fit = cf_fit.reshape([cf_fit.shape[0], -1])
        cf_mod = jl_poly(np.array([wfe_drift]), cf_fit, use_legendre=True, lxmap=lxmap)
        cf_mod = cf_mod.reshape(cf_fit_shape[1:])

        # Pad cf_mod array with 0s if undersized
        if not np.allclose(psf_coeff.shape, cf_mod.shape):
            new_shape = psf_coeff.shape[1:]
            cf_mod_resize = np.array([pad_or_cut_to_size(im, new_shape) for im in cf_mod])
            cf_mod = cf_mod_resize
    
    return cf_mod


def _coeff_mod_wfe_field(self, coord_vals, coord_frame):
    """
    Modify PSF polynomial coefficients as a function of V2/V3 position.
    """

    v2 = v3 = None
    cf_mod = 0
    nfield = 1

    psf_coeff_hdr = self.psf_coeff_header
    psf_coeff     = self.psf_coeff

    cf_fit = self._psf_coeff_mod['si_field'] 
    v2grid  = self._psf_coeff_mod['si_field_v2grid'] 
    v3grid  = self._psf_coeff_mod['si_field_v3grid'] 

    # Modify PSF coefficients based on position
    if coord_vals is None:
         pass
    elif self._psf_coeff_mod['si_field'] is None:
        _log.warning("You must run `gen_wfefield_coeff` first before setting the coord_vals parameter.")
        _log.warning("`calc_psf_from_coeff` will continue with default PSF.")
        cf_mod = 0
    else:
        # Determine V2/V3 coordinates
        cframe = coord_frame.lower()
        if cframe=='tel':
            v2, v3 = coord_vals
            v2, v3 = (v2/60., v3/60.) # convert to arcmin
        elif cframe in ['det', 'sci', 'idl']:
            x = np.array(coord_vals[0])
            y = np.array(coord_vals[1])
            
            try:
                apname = self.aperturename
                siaf_ap = self.siaf[apname]
                v2, v3 = siaf_ap.convert(x,y, cframe, 'tel')
                v2, v3 = (v2/60., v3/60.) # convert to arcmin
            except: 
                # apname = self.get_siaf_apname()
                self._update_aperturename()
                apname = self.aperturename
                if apname is None:
                    _log.warning('No suitable aperture name defined to determine V2/V3 coordiantes')
                else:
                    _log.warning('self.siaf_ap not defined; assuming {}'.format(apname))
                    siaf_ap = self.siaf[apname]
                    v2, v3 = siaf_ap.convert(x,y, cframe, 'tel')
                    v2, v3 = (v2/60., v3/60.)
                _log.warning('Update self.siaf_ap for more specific conversions to V2/V3.')
        else:
            _log.warning("coord_frame setting '{}' not recognized.".format(coord_frame))
            _log.warning("`calc_psf_from_coeff` will continue with default PSF.")

    # PSF Modifications assuming we successfully found v2/v3
    if (v2 is not None):
        # print(v2,v3)
        nfield = np.size(v2)
        cf_mod = field_coeff_func(v2grid, v3grid, cf_fit, v2, v3)

        # Pad cf_mod array with 0s if undersized
        psf_cf_dim = len(psf_coeff.shape)
        if not np.allclose(psf_coeff.shape, cf_mod.shape[-psf_cf_dim:]):
            new_shape = psf_coeff.shape[1:]
            cf_mod_resize = np.array([pad_or_cut_to_size(im, new_shape) for im in cf_mod])
            cf_mod = cf_mod_resize

    return cf_mod, nfield