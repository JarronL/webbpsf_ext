import numpy as  np
import os

import matplotlib
from matplotlib import pyplot as plt

import astropy.units as u
from astropy.io import fits, ascii
from astropy.config import ConfigItem

import stsynphot as stsyn
import synphot
from synphot.units import validate_wave_unit, convert_flux
from synphot.models import get_waveset

# Extend default wavelength range to 35 um
_wave, _wave_str = synphot.generate_wavelengths(
        minwave=500, maxwave=350000, num=10000, delta=None, log=False,
        wave_unit='angstrom')
stsyn.conf.waveset = _wave_str
stsyn.conf.waveset_array = _wave.to_value('angstrom').tolist()
# JWST 25m^2 collecting area
# Flux loss from masks and occulters are taken into account in WebbPSF
stsyn.conf.area = 25.78e4

##########################################################
# Download synphot data files
##########################################################

# CDBS directory from environment variable
def download_cdbs_data(cdbs_path=None, verbose=False):

    from synphot.utils import download_data

    if cdbs_path is None:
        cdbs_path = os.environ.get('PYSYN_CDBS', None)
        if cdbs_path is None:
            raise ValueError("Environment variable PYSYN_CDBS is not set.")

    # Download synphot data files
    res = download_data(cdbs_path, verbose=False)

    if len(res) > 0 and verbose:
        for r in res:
            print(f'Downloaded: {r}')


##########################################################
# Add some units to astropy.units
##########################################################

def validate_unit(input_unit):
    """Add lowercase jy to synphot units validation function"""

    if isinstance(input_unit, str):
        if input_unit in ['njy', 'nanojy']:
            output_unit = u.nJy
        elif input_unit in ['ujy', 'mujy', 'microjy']:
            output_unit = u.uJy
        elif input_unit=='mjy':
            output_unit = u.mJy
        elif input_unit=='jy':
            output_unit = u.Jy
        elif input_unit=='Mjy':
            output_unit = u.MJy
        elif input_unit=='counts':
            output_unit = u.count
        else:
            output_unit = synphot.units.validate_unit(input_unit)
    else:
        output_unit = synphot.units.validate_unit(input_unit)

    return output_unit
        
##########################################################
# Bandpass classes and convenience functions
##########################################################

class Bandpass(synphot.SpectralElement):
    """ A class representing a bandpass filter.

    Parameters:
    ----------
    modelclass : cls
        Model class from `astropy.modeling`.
    waveunits : str, optional
        Units of the wavelength, default is 'angstrom'.
    name : str, optional
        Name of the bandpass, default is 'UnnamedBandpass'.
    kwargs : dict
        Model parameters accepted by ``modelclass``. Each parameter can
        be either a Quantity or number. If the latter, assume pre-defined
        internal units.
    """

    def __init__(self, modelclass, waveunits='angstrom', name=None, **kwargs):

        # Initialize parent class
        super().__init__(modelclass, **kwargs)

        # Set user wavelength units
        self._waveunits = validate_wave_unit(waveunits)

        if name is not None:
            self.name = name

    @property
    def waveset(self):
        """Optimal wavelengths for sampling the spectrum or bandpass."""

        w = get_waveset(self.model)
        if w is None:
            # Get default waveset from stsynphot
            w = stsyn.conf.waveset_array
        self._validate_wavelengths(w)
        return w * self._internal_wave_unit

    @property
    def waveunits(self):
        """ User wavelength units """
        return self._waveunits
    
    @property
    def wave(self):
        """ User wavelength array """
        return self.waveset.to_value(self.waveunits)
    
    @property
    def throughput(self):
        """ Throughput array """
        return self(self.waveset).value
    
    @property
    def name(self):
        """ Bandpass name """
        return self.meta.get('expr', 'UnnamedBandpass')
    @name.setter
    def name(self, value):
        self.meta['expr'] = value
    
    def convert(self, new_waveunits):
        """ Convert wavelength units """
        self._waveunits = validate_wave_unit(new_waveunits)

    def filter_width(self):
        """ Calculate the width of the filter at half max """
        from .bandpasses import filter_width
        # Wavelength range of filter in microns
        wmin, wmax = filter_width(self)
        dw = (wmax - wmin) * u.um
        return dw.to(self._internal_wave_unit)
    
    def unit_response(self, area=None, wavelengths=None):
        if area is None:
            area = stsyn.conf.area
        return super().unit_response(area, wavelengths=wavelengths)
    
    def resample(self, new_wave):
        """ Resample the bandpass to a new wavelength array """
        throughput = self(new_wave).value
        return ArrayBandpass(new_wave, throughput, name=self.name)

class BoxFilter(Bandpass):
    """ Box filter with a given center and width
    
    Parameters
    ----------
    center : float or `astropy.units.Quantity`
        Center of the box filter. Can be a number or astropy units Quantity.
    width : float or `astropy.units.Quantity`
        Width of the box filter. Can be a number or astropy units Quantity.
    waveunits : str, optional
        Units of the wavelength, default is 'angstrom'.
    name : str, optional
        Name of the bandpass. Will default to 'Box at {center} ({width} wide)'.
    """

    def __init__(self, center, width, waveunits='angstrom', **kwargs):

        from synphot.models import Box1D

        # Check if center and width are astropy units
        wunits = validate_wave_unit(waveunits)
        center = center.to_value(wunits) if isinstance(center, u.Quantity) else center * wunits
        width  = width.to_value(wunits)  if isinstance(width, u.Quantity)  else width * wunits

        # Initialize parent class
        super().__init__(Box1D, x_0=center, width=width, waveunits=waveunits, **kwargs)

    @property
    def name(self):
        """ Bandpass name """
        center = self.model.x_0.value * self._internal_wave_unit
        width  = self.model.width.value * self._internal_wave_unit
        name = f'Box at {center} ({width} wide)'
        return self.meta.get('expr', name)
    @name.setter
    def name(self, value):
        self.meta['expr'] = value

    def filter_width(self):
        """ Calculate the width of the filter at half max """
        return self.model.width.value * self._internal_wave_unit
    
    def taper(self, wavelengths=None):
        """Taper the spectrum or bandpass.

        The wavelengths to use for the first and last points are
        calculated by using the same ratio as for the 2 interior points.

        Parameters
        ----------
        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for tapering.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, `waveset` is used.

        Returns
        -------
        sp : `BaseSpectrum`
            Tapered empirical spectrum or bandpass.
            ``self`` is returned if already tapered (e.g., box model).

        """
        x = self._validate_wavelengths(wavelengths)

        # Calculate new end points for tapering
        w1 = x[0] ** 2 / x[1]
        w2 = x[-1] ** 2 / x[-2]

        # Special handling for empirical data.
        # This is to be compatible with ASTROLIB PYSYNPHOT behavior.
        if isinstance(self._model, synphot.Empirical1D):
            y1 = self._model.lookup_table[0]
            y2 = self._model.lookup_table[-1]
        # Other models can just evaluate at new end points
        else:
            y1 = self(w1)
            y2 = self(w2)

        # Nothing to do
        if y1 == 0 and y2 == 0:
            return self  # Do we need a deepcopy here?

        y = self(x)

        if y1 != 0:
            x = np.insert(x, 0, w1)
            y = np.insert(y, 0, 0.0 * y.unit)
        if y2 != 0:
            x = np.insert(x, x.size, w2)
            y = np.insert(y, y.size, 0.0 * y.unit)

        name = f'{self.name} tapered'

        return Bandpass(synphot.Empirical1D, points=x, lookup_table=y, name=name)
    
    @plt.style.context('webbpsf_ext.wext_style')
    def plot(self, wavelengths, **kwargs):

        super().plot(wavelengths, **kwargs)
    
class UniformTransmission(Bandpass):
    
    def __init__(self, amplitude, waveunits='angstrom', name=None, **kwargs):

        from astropy.modeling.models import Const1D

        # Initialize parent class
        super().__init__(Const1D, amplitude=amplitude, waveunits=waveunits, **kwargs)
    
    @property
    def name(self):
        """ Bandpass name """
        amplitude  = self.model.amplitude.value
        name = f'Flat ({amplitude})'
        return self.meta.get('expr', name)
    @name.setter
    def name(self, value):
        self.meta['expr'] = value

    def filter_width(self):
        """ Calculate the width of the filter at half max """
        return None

def ObsBandpass(filtername):

    # check if 'bessel', 'johnson', or 'cousins' string exist in filtername
    filtername_lower = filtername.lower()
    if np.any([s in filtername_lower for s in ['bessel', 'johnson', 'cousins']]):
        return Bandpass.from_filter(filtername_lower.replace(',', '_'))
    else:
        raise ValueError(f'{filtername} not a valid bandpass. If using HST filters, use stsynphot package.')

def ArrayBandpass(wave, throughput, name="UnnamedArrayBandpass", keep_neg=False, **kwargs):
    """ Generate a synphot bandpass from arrays

    Parameters
    ----------
    wave, throughput : array_like
        Wavelength and throughput arrays.
    name : str
        Description of the spectrum. Default is "UnnamedArrayBandpass".
    keep_neg : bool, optional
        Whether to keep negative throughput values. Default is False.

    Keyword Args
    ------------
    waveunits : str
        Wavelength unit of ``wave`` input. Default is Angstrom.
    """

    # Check if waveunits have been specified
    waveunits = kwargs.pop('waveunits', None)

    if isinstance(wave, u.Quantity):
        waveunits = wave.unit.to_string() if waveunits is None else waveunits
    else:
        waveunits = 'angstrom' if waveunits is None else waveunits
        wave = wave * validate_unit(waveunits)

    return Bandpass(synphot.Empirical1D, points=wave, lookup_table=throughput, 
                    name=name, waveunits=waveunits, keep_neg=keep_neg, **kwargs)


def FileBandpass(filename, keep_neg=True, **kwargs):
    """ Create synphot bandpass from file.

    If filename has 'fits' or 'fit' suffix, it is read as FITS.
    Otherwise, it is read as ASCII.

    Parameters
    ----------
    filename : str
        Name of file containing bandpass information. File must have
        two columns: wavelength and throughput.
    keep_neg : bool, optional
        Whether to keep negative throughput values. Default is True.

    Keyword Args
    ------------
    waveunits : str, optional
        Units of wavelength in file. Default is assumed to be Angstrom.
    flux_col : str, optional
        Name of the column containing the throughput values. If not provided,
        the default column name 'THROUGHPUT' is used for FITS files.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `read_spec` function.

    Returns
    -------
    Bandpass
        Bandpass object containing the wavelength and throughput data.

    Examples
    --------
    >>> bandpass = FileBandpass('bandpass.txt', waveunits='nm')
    """

    from synphot import specio

    # Check if waveunits have been specified
    waveunits = kwargs.pop('waveunits', 'angstrom')

    # Pass wave_unit and flux_unit to specio.read_spec
    kwargs['wave_unit'] = validate_unit(waveunits)
    kwargs['flux_unit'] = Bandpass._internal_flux_unit

    if filename.endswith('fits') or filename.endswith('fit'):
        kwargs['flux_col'] = kwargs.get('flux_col', 'THROUGHPUT')

    header, wave, th = specio.read_spec(filename, **kwargs)
    name = kwargs.get('name', os.path.basename(filename))

    return ArrayBandpass(wave, th, waveunits=waveunits, 
                         name=name, keep_neg=keep_neg, meta={'header': header})



##########################################################
# Spectra classes and convenience functions
##########################################################

class Spectrum(synphot.SourceSpectrum):
    """ A class representing a spectrum.

    Parameters:
    ----------
    *args : tuple
        Positional arguments passed to the parent class.
    waveunits : str, optional
        Units of the wavelength, default is 'angstrom'.
    fluxunits : str, optional
        Units of the flux, default is 'flam'.
    name : str, optional
        Name of the spectrum, default is 'UnnamedSpectrum'.
    **kwargs : dict
        Keyword arguments passed to the parent class.
    """

    def __init__(self, modelclass, waveunits='angstrom', fluxunits='photlam', name=None, **kwargs):

        # Initialize parent class
        super().__init__(modelclass, **kwargs)

        # Set user wavelength and flux units
        self._waveunits = validate_unit(waveunits)
        self._fluxunits = validate_unit(fluxunits)

        if name is not None:
            self.name = name

    @property
    def waveset(self):
        """Optimal wavelengths for sampling the spectrum."""

        w = get_waveset(self.model)
        if w is None:
            # Get default waveset from stsynphot
            w = stsyn.conf.waveset_array
        self._validate_wavelengths(w)
        return w * self._internal_wave_unit

    @property
    def waveunits(self):
        """ User wavelength units """
        return self._waveunits
    
    @property
    def fluxunits(self):
        """ User flux units """
        return self._fluxunits
    
    @property
    def wave(self):
        """ User wavelength array """
        return self.waveset.to_value(self.waveunits)
    
    @property
    def flux(self):
        """ User flux array """
        flux_intrinsic = self(self.waveset)
        if self._fluxunits == self._internal_flux_unit:
            return flux_intrinsic.value
        else:
            flux_output = convert_flux(self.waveset, flux_intrinsic, self.fluxunits)
            return flux_output.value
    
    @property
    def name(self):
        """ Spectrum name """
        return self.meta.get('expr', 'UnnamedSpectrum')
    @name.setter
    def name(self, value):
        self.meta['expr'] = value

    def convert(self, new_units):
        """ Convert wavelength or flux units """
        from synphot.exceptions import SynphotError

        try:
            self._waveunits = validate_wave_unit(new_units)
        except SynphotError:
            self._fluxunits = validate_unit(new_units)

    def renorm(self, RNval, RNUnits, band, force=False):
        """Renormalize the spectrum to the specified value, unit, and bandpass.

        This wraps `normalize` attribute for convenience using the telescope
        area defined in stsynphot configuration (defaulted to JWST collecting
        area).

        Parameters
        ----------
        RNval : float or astropy.units.Quantity
            Flux value for renormalization.
        RNUnits : str or astropy.units.Unit
            Flux unit for renormalization. If ``RNval`` is a Quantity,
            this is ignored.
        band : `Bandpass`
            Bandpass to renormalize in.
        force : bool
            Force renormalization regardless of overlap status with given
            bandpass. If `True`, overlap check is skipped. Default is `False`.
        """
            
        # Check if RNval is a Quantity
        if not isinstance(RNval, u.Quantity):
            RNUnits = validate_unit(RNUnits)
            RNval = RNval * RNUnits

        # Renormalize
        return self.normalize(RNval, band, area=stsyn.conf.area, 
                              vegaspec=stsyn.Vega, force=force)


def ArraySpectrum(wave, flux, name="UnnamedArraySpectrum", keep_neg=False, **kwargs):
    """ Generate a synphot spectrum from arrays

    Parameters
    ----------
    wave, flux : array_like
        Wavelength and flux arrays.
    name : str
        Description of the spectrum. Default is "UnnamedArraySpectrum".
    keep_neg : bool, optional
        Whether to keep negative throughput values. Default is False.

    Keyword Args
    ------------
    waveunits : str
        Wavelength unit of ``wave`` input. Default is Angstrom.
    fluxunits : str
        Flux unit of ``flux`` input. Default is photlam.
    """

    # Check if waveunits have been specified
    waveunits = kwargs.pop('waveunits', None)
    if isinstance(wave, u.Quantity):
        waveunits = wave.unit.to_string() if waveunits is None else waveunits
    else:
        waveunits = 'angstrom' if waveunits is None else waveunits
        wave = wave * validate_unit(waveunits)

    # Check if fluxunits have been specified
    fluxunits = kwargs.pop('fluxunits', None)
    if isinstance(flux, u.Quantity):
        fluxunits = flux.unit.to_string() if fluxunits is None else fluxunits
    else:
        fluxunits = 'photlam' if fluxunits is None else fluxunits
        flux = flux * validate_unit(fluxunits)

    return Spectrum(synphot.models.Empirical1D, points=wave, lookup_table=flux, 
                    name=name, waveunits=waveunits, fluxunits=fluxunits,
                    keep_neg=keep_neg, **kwargs)

def FileSpectrum(filename, keep_neg=False, **kwargs):
    """ Create synphot spectrum from file.

    If filename has 'fits' or 'fit' suffix, it is read as FITS.
    Otherwise, it is read as ASCII.

    Parameters
    ----------
    filename : str
        Name of file containing spectral information. File must have
        two columns: wavelength and throughput.
    keep_neg : bool, optional
        Whether to keep negative throughput values. Default is False.

    Keyword Args
    ------------
    waveunits : str, optional
        Units of wavelength in file. Default is assumed to be Angstrom.
    fluxunits : str, optional
        Units of flux in file. Default is assumed to be PHOTLAM.
    wave_col : str, optional
        Name of the column containing the wavelength values. If not provided,
        the default column name 'WAVELENGTH' is used for FITS files.
    flux_col : str, optional
        Name of the column containing the flux values. If not provided,
        the default column name 'FLUX' is used for FITS files.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `read_spec` function.

    Returns
    -------
    Spectrum
        Spectrum object containing the wavelength and throughput data.

    Examples
    --------
    >>> sp = FileSpectrum('spectrum.txt', waveunits='nm', fluxunits='mjy')
    """

    from synphot import specio

    # Check if waveunits have been specified
    waveunits = kwargs.pop('waveunits', 'angstrom')
    fluxunits = kwargs.pop('fluxunits', 'photlam')

    # Pass wave_unit and flux_unit to specio.read_spec
    kwargs['wave_unit'] = validate_unit(waveunits)
    kwargs['flux_unit'] = validate_unit(fluxunits)

    header, wave, flux = specio.read_spec(filename, **kwargs)
    name = kwargs.get('name', os.path.basename(filename))

    return ArraySpectrum(wave, flux, waveunits=waveunits, fluxunits=fluxunits,
                         name=name, keep_neg=keep_neg, meta={'header': header})

def Icat(gridname, t_eff, metallicity, log_g, name=None, **kwargs):

    sp = stsyn.grid_to_spec(gridname, t_eff, metallicity, log_g)
    wave = sp.waveset
    flux = sp(wave)
    meta = sp.meta

    return ArraySpectrum(wave, flux, name=name, meta=meta, **kwargs)

def FlatSpectrum(flux, name='UnnamedFlatSpectrum', **kwargs):
    """ Create a flat spectrum.

    Creates a flat spectrum with the specified flux density amplitude.
    The wavelength range is set to the stsynphot default range.

    Parameters
    ----------
    flux : float or `astropy.units.Quantity`
        Flux density amplitude of flat spectrum. Can be a number or
        astropy units Quantity.
    name : str
        Description of the spectrum. Default is "UnnamedFlatSpectrum".

    Keyword Args
    ------------
    waveunits : str
        Wavelength unit of ``wave`` input. Default is Angstrom.
    fluxunits : str
        Flux unit of ``flux`` input. Default is photlam.
    **kwargs : dict, optional
        Additional keyword arguments to be passed to the `Spectrum` function.

    Returns
    -------
    Spectrum
        Flat spectrum object.
    """

    from synphot.models import ConstFlux1D

    # Check if fluxunits have been specified
    fluxunits = kwargs.pop('fluxunits', None)
    if isinstance(flux, u.Quantity):
        fluxunits = flux.unit.to_string() if fluxunits is None else fluxunits
    else:
        fluxunits = 'photlam' if fluxunits is None else fluxunits
        flux = flux * validate_unit(fluxunits)

    return Spectrum(ConstFlux1D, amplitude=flux, fluxunits=fluxunits, name=name, **kwargs)


def load_vega(vegafile=None, **kwargs):
    """Convenience function to load Vega spectrum

    Parameters
    ----------
    vegafile : str or `None`, optional
        Vega spectrum filename.
        If `None`, use ``synphot.config.conf.vega_file``.

    kwargs : dict
        Keywords acceptable by :func:`synphot.specio.read_remote_spec`.

    Returns
    -------
    sp : `synphot.spectrum.SourceSpectrum` or `None`
        Vega spectrum. `None` if failed.

    """
    from synphot.config import conf as synconf
    import warnings
    from astropy.utils.exceptions import AstropyUserWarning

    if vegafile is None:
        vegafile = synconf.vega_file

    with synconf.set_temp('vega_file', vegafile):
        try:
            Vega = Spectrum.from_vega(**kwargs)
        except Exception as e:
            Vega = None
            warnings(
                f'Failed to load Vega spectrum from {vegafile}; Functionality '
                f'involving Vega will be cripped: {repr(e)}',
                AstropyUserWarning)
            
    return Vega

def BlackBody(temperature):
    """Reproduces pysynphot blackbody function"""
    return Spectrum(synphot.models.BlackBodyNorm1D, temperature=temperature)

# Load default Vega
Vega = load_vega(encoding='binary')

##########################################################
# Observation class
##########################################################

class Observation(synphot.Observation):

    area = stsyn.conf.area

    def __call__(self, wavelengths, flux_unit=None):
        """Sample the spectrum.

        Parameters
        ----------
        wavelengths : array-like or `~astropy.units.quantity.Quantity`
            Wavelength values for sampling. If not a Quantity,
            assumed to be in Angstrom.

        flux_unit : str, `~astropy.units.Unit`, or `None`
            Flux is converted to this unit.
            If not given, internal unit is used.

        kwargs : dict
            Keywords acceptable by :func:`~synphot.units.convert_flux`.

        Returns
        -------
        sampled_result : `~astropy.units.quantity.Quantity`
            Sampled flux in the given unit.
            Might have negative values.

        """
        kwargs = {'flux_unit': flux_unit, 'area': self.area, 'vegaspec': stsyn.Vega}
        return super().__call__(wavelengths, **kwargs)


    def __init__(self, spec, band, binset=None, force='none', waveunits='angstrom', fluxunits='photlam', name=None):

        # Initialize parent class
        super().__init__(spec, band, binset=binset, force=force)

        # Set user wavelength and flux units
        self._waveunits = validate_unit(waveunits)
        self._fluxunits = validate_unit(fluxunits)

        if name is not None:
            self.name = name

    @property
    def name(self):
        """ Spectrum name """
        return self.meta.get('expr', 'UnnamedSpectrum')
    @name.setter
    def name(self, value):
        self.meta['expr'] = value

    @property
    def waveset(self):
        """Optimal wavelengths for sampling the spectrum."""

        w = get_waveset(self.model)
        if w is None:
            # Get default waveset from stsynphot
            w = stsyn.conf.waveset_array
        self._validate_wavelengths(w)
        return w * self._internal_wave_unit

    @property
    def waveunits(self):
        """ User wavelength units """
        return self._waveunits
    
    @property
    def binwave(self):
        """ User binned wavelength array """
        return self.binset.to_value(self.waveunits)
    
    @property
    def fluxunits(self):
        """ User flux units """
        return self._fluxunits
    
    @property
    def wave(self):
        """ User wavelength array """
        return self.waveset.to_value(self.waveunits)
    
    @property
    def flux(self):
        """ User flux array """
        flux_intrinsic = self(self.waveset)
        if self._fluxunits == self._internal_flux_unit:
            return flux_intrinsic.value
        else:
            flux_output = convert_flux(self.waveset, flux_intrinsic, self.fluxunits,
                                       area=self.area, vegaspec=stsyn.Vega)
            return flux_output.value
    
    @property
    def name(self):
        """ Spectrum name """
        return self.meta.get('expr', 'UnnamedSpectrum')
    @name.setter
    def name(self, value):
        self.meta['expr'] = value

    def convert(self, new_units):
        """ Convert wavelength or flux units """
        from synphot.exceptions import SynphotError

        try:
            self._waveunits = validate_wave_unit(new_units)
        except:
            self._fluxunits = validate_unit(new_units)

    def sample_binned(self, wavelengths=None, flux_unit=None, **kwargs):
        kwargs['area'] = kwargs.get('area', self.area)
        kwargs['vegaspec'] = kwargs.get('vegaspec', stsyn.Vega)
        if flux_unit == 'counts':
            flux_unit = 'count'
        return super().sample_binned(wavelengths=wavelengths, flux_unit=flux_unit, **kwargs)

    def efflam(self):
        """ Effective wavelength """
        return self.effective_wavelength().to_value(self.waveunits)

    def effstim(self, flux_unit=None, wavelengths=None, 
                area=None, vegaspec=stsyn.Vega):
        """Calculate effective stimulus for given flux unit.

        Area is set to the default value in stsynphot configuration (JWST).

        Parameters
        ----------
        flux_unit : str or `~astropy.units.Unit` or `None`
            The unit of effective stimulus.
            COUNT gives result in count/s (see :meth:`countrate` for more
            options).
            If not given, internal unit is used.

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling. This must be given if
            ``self.waveset`` is undefined for the underlying spectrum model(s).
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` is used.

        Returns
        -------
        eff_stim : float
            Observation effective stimulus based on given flux unit.
        """
        area = self.area if area is None else area
        if flux_unit == 'counts':
            flux_unit = 'count'
        res = super().effstim(flux_unit=flux_unit, wavelengths=wavelengths,
                              area=area, vegaspec=vegaspec)
        
        try:
            return res.value
        except AttributeError:
            return res
        
    def countrate(self, area=None, binned=True, wavelengths=None, waverange=None,
                  force=False):
        """Calculate :ref:`effective stimulus <synphot-formula-effstim>`
        in count/s.

        Parameters
        ----------
        binned : bool
            Sample data in native wavelengths if `False`.
            Else, sample binned data (default).

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling. This must be given if
            ``self.waveset`` is undefined for the underlying spectrum model(s).
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` or `binset` is used, depending
            on ``binned``.

        waverange : tuple of float, Quantity, or `None`
            Lower and upper limits of the desired wavelength range.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, the full range is used.

        force : bool
            If a wavelength range is given, partial overlap raises
            an exception when this is `False` (default). Otherwise,
            it returns calculation for the overlapping region.
            Disjoint wavelength range raises an exception regardless.

        Returns
        -------
        count_rate : float
            Observation effective stimulus in count/s.

        Raises
        ------
        synphot.exceptions.DisjointError
            Wavelength range does not overlap with observation.

        synphot.exceptions.PartialOverlap
            Wavelength range only partially overlaps with observation.

        synphot.exceptions.SynphotError
            Calculation failed, including but not limited to NaNs in flux.

        """
        area = self.area if area is None else area
        res = super().countrate(area, binned=binned, wavelengths=wavelengths,
                                waverange=waverange, force=force)
        return res.value
    
    @plt.style.context('webbpsf_ext.wext_style')
    def plot(self, binned=True, wavelengths=None, flux_unit=None, **kwargs): 
        """Plot the observation.

        .. note:: Uses ``matplotlib``.

        Parameters
        ----------
        binned : bool
            Plot data in native wavelengths if `False`.
            Else, plot binned data (default).

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` or `binset` is used, depending
            on ``binned``.

        flux_unit : str or `~astropy.units.Unit` or `None`
            Flux is converted to this unit for plotting.
            If not given, internal unit is used.

        kwargs : dict
            See :func:`synphot.spectrum.BaseSpectrum.plot`.

        Raises
        ------
        synphot.exceptions.SynphotError
            Invalid inputs.

        """

        kwargs['area'] = kwargs.get('area', self.area)
        kwargs['vegaspec'] = kwargs.get('vegaspec', stsyn.Vega)        
        if flux_unit == 'counts':
            flux_unit = 'count'
        return super().plot(binned=binned, wavelengths=wavelengths, 
                            flux_unit=flux_unit, **kwargs)

    def as_spectrum(self, binned=True, wavelengths=None):
        """Reduce the observation to an empirical source spectrum.

        An observation is a complex object with some restrictions
        on its capabilities. At times, it would be useful to work
        with the observation as a simple object that is easier to
        manipulate and takes up less memory.

        This is also useful for writing an observation as sampled
        spectrum out to a FITS file.

        Parameters
        ----------
        binned : bool
            Write out data in native wavelengths if `False`.
            Else, write binned data (default).

        wavelengths : array-like, `~astropy.units.quantity.Quantity`, or `None`
            Wavelength values for sampling.
            If not a Quantity, assumed to be in Angstrom.
            If `None`, ``self.waveset`` or `binset` is used, depending
            on ``binned``.

        Returns
        -------
        sp : `~synphot.spectrum.SourceSpectrum`
            Empirical source spectrum.

        """
        from synphot import Empirical1D

        if binned:
            w, y = self._get_binned_arrays(
                wavelengths, self._internal_flux_unit)
        else:
            w, y = self._get_arrays(
                wavelengths, flux_unit=self._internal_flux_unit)

        header = {'observation': str(self), 'binned': binned}
        return Spectrum(Empirical1D, points=w, lookup_table=y, name=self.name,
                        meta={'header': header})
    
def Extinction(ebv, name):
    """ Create a synphot extinction object

    Parameters
    ----------
    ebv : float
        E(B-V) value of extinction.
    name : str
        Extinction model name. Choose from: 'lmc30dor', 'lmcavg', 'mwavg',
        'mwdense', 'mwrv21', 'mwrv40', 'smcbar', or 'xgalsb'.

    Returns
    -------
    Extinction
        Extinction object.
    """

    if name=='mwrv4':
        name = 'mwrv40'

    return stsyn.ebmvx(name, ebv)

