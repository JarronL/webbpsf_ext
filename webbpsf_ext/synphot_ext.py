import numpy as  np
import os

import astropy.units as u
from astropy.io import fits, ascii
from astropy.config import ConfigItem

import stsynphot as stsyn
import synphot
from synphot.units import validate_wave_unit, convert_flux

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

    if isinstance(input_unit, str) and 'jy' in input_unit:
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
        from synphot.models import get_waveset
        from synphot.utils import validate_wavelengths

        w = get_waveset(self.model)
        if w is None:
            # Get default waveset from stsynphot
            w = stsyn.conf.waveset_array
        validate_wavelengths(w)
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

    def __init__(self, center, width, waveunits='angstrom', name=None, **kwargs):

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

# def ObsBandpass

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

    return Bandpass(synphot.models.Empirical1D, points=wave, lookup_table=throughput, 
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
        """Optimal wavelengths for sampling the spectrum or bandpass."""
        from synphot.models import get_waveset
        from synphot.utils import validate_wavelengths

        w = get_waveset(self.model)
        if w is None:
            # Get default waveset from stsynphot
            w = stsyn.conf.waveset_array
        validate_wavelengths(w)
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


def ArraySpectrum(wave, flux, name="UnnamedArraySpectrum", keep_neg=False, **kwargs):
    """ Generate a synphot bandpass from arrays

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
    """ Create synphot bandpass from file.

    If filename has 'fits' or 'fit' suffix, it is read as FITS.
    Otherwise, it is read as ASCII.

    Parameters
    ----------
    filename : str
        Name of file containing bandpass information. File must have
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

    return ArrayBandpass(wave, flux, waveunits=waveunits, fluxunits=fluxunits,
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
