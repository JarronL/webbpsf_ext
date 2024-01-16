import pytest
import numpy as np
import os

from astropy.io import ascii

from webbpsf_ext import synphot_ext, bandpasses

from synphot.tests import test_spectrum_bandpass
from synphot import GaussianAbsorption1D, Empirical1D

from astropy.utils.data import get_pkg_data_filename
import astropy.units as u

# Option to skip tests if pysynphot is not installed
try:
    import pysynphot
    pysynphot_installed = True
except ImportError: 
    pysynphot_installed = False

@pytest.mark.skipif(not pysynphot_installed, reason="pysynphot is not installed")
def test_bandpass_equivalence():
    """
    Test the equivalence of bandpasses between pysynphot and synphot_convert.

    This function compares the wave and throughput arrays of a bandpass created using
    pysynphot and synphot_convert. It checks if the arrays are equal and raises an
    assertion error if they are not.

    Note: This test requires the pysynphot package to be installed. If it is not
    installed, the test will be skipped.
    """

    bp_2mass_dir = os.path.join(bandpasses._bp_dir, '2MASS')
    file = '2mass_ks.txt'
    name='Ks-Band'
    tbl = ascii.read(os.path.join(bp_2mass_dir, file), names=['Wave', 'Throughput'])
    wave, th = tbl['Wave'].data*1e4, tbl['Throughput'].data

    # Create a pysynphot.ObsBandpass object
    pysyn_bandpass = pysynphot.ArrayBandpass(wave, th, name=name)
    pysyn_bandpass.convert('um')

    # Create a synphot_convert.Bandpass object
    synphot_bandpass = synphot_ext.ArrayBandpass(wave, th, name=name)
    synphot_bandpass.convert('um')

    # Compare the wave and throughput arrays
    pysyn_wave = pysyn_bandpass.wave
    synphot_wave = synphot_bandpass.wave
    pysyn_throughput = pysyn_bandpass.throughput
    synphot_throughput = synphot_bandpass.throughput

    # Check if the wave and throughput arrays are close
    assert np.allclose(pysyn_wave, synphot_wave)
    assert np.allclose(pysyn_throughput, synphot_throughput)

def test_validate_unit():
    
    # Test lowercase jy units
    unit_strings = ['jy', 'njy', 'ujy', 'mjy', 'Mjy']
    for unit_str in unit_strings:
        synphot_ext.validate_unit(unit_str)

    # Test that Jys fails
    with pytest.raises(ValueError):
        synphot_ext.validate_unit('Jys')

@pytest.mark.parametrize(
    'filtername',
    ['bessel_j', 'bessel_h', 'bessel_k', 'cousins_r', 'cousins_i',
     'johnson_u', 'johnson_b', 'johnson_v', 'johnson_r', 'johnson_i',
     'johnson_j', 'johnson_k'])
def test_filter(filtername):
    """Test loading pre-defined bandpass.

    .. note::

        Filter data quality is not checked as it depends on the file.

    """

    bp = synphot_ext.ObsBandpass(filtername)
    assert isinstance(bp.model, Empirical1D)
    assert filtername in bp.meta['expr']

    bp = synphot_ext.ObsBandpass(filtername.replace('_', ','))
    assert isinstance(bp.model, Empirical1D)
    assert filtername in bp.meta['expr']


class TestBandpassFromFile(test_spectrum_bandpass.TestEmpiricalBandpassFromFile):

    def setup_class(self):
        bandfile = get_pkg_data_filename(
            os.path.join('data', 'hst_acs_hrc_f555w.fits'),
            package='synphot.tests')
        self.bp = synphot_ext.Bandpass.from_file(bandfile)

    def test_filter_width(self):
        """Test filter width calculation."""
        assert np.allclose(self.bp.filter_width(), 1248.8194*u.AA, rtol=2.5e-5)

    def test_convert(self):
        """Test wavelength conversion."""
        self.bp.convert('um')
        assert self.bp.waveunits == u.um
        assert np.allclose(self.bp.wave, self.bp.waveset.to_value(u.um))
        self.bp.convert(self.bp._internal_wave_unit)

    def test_name_setter(self):
        """Test setting name."""
        self.bp.name = 'foo'
        assert self.bp.name == 'foo'
        assert self.bp.meta['expr'] == 'foo'


class TestBoxFilter(test_spectrum_bandpass.TestBoxBandpass):
    def setup_class(self):
        self.bp = synphot_ext.BoxFilter(5000, 100)

    def test_filter_width(self):
        """Test filter width calculation."""
        assert np.allclose(self.bp.filter_width(), 100*u.AA)


class TestBuildModelsBandpass:
    """Test compatiblity with other models not tested above."""
    def test_GaussianAbsorption1D(self):
        """This should be unitless, not a source spectrum."""
        bp = synphot_ext.Bandpass(
            GaussianAbsorption1D, amplitude=0.8, mean=5500, stddev=50)
        y = bp([5300, 5500, 5700])
        assert np.allclose(y.value, [0.99973163, 0.2, 0.99973163])
