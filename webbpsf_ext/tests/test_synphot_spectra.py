import pytest
import numpy as np
import os

from webbpsf_ext import synphot_ext

from synphot import units
from synphot import BlackBodyNorm1D, GaussianFlux1D, PowerLawFlux1D, ConstFlux1D
from synphot.exceptions import SynphotError
from synphot.tests import test_spectrum_source
from synphot.tests.test_units import (
    _area, _wave, _flux_jy, _flux_photlam, _flux_vegamag
)

from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.data import get_pkg_data_filename
import astropy.units as u


def test_load_vspec():
    """Load VEGA spectrum once here to be used later."""
    global _vspec2
    _vspec2 = synphot_ext.Spectrum.from_vega()

@pytest.mark.parametrize(
    ('in_q', 'out_u', 'ans'),
    [(_flux_photlam, units.VEGAMAG, _flux_vegamag),
     (_flux_vegamag, units.PHOTLAM, _flux_photlam),
     (_flux_jy, units.VEGAMAG, _flux_vegamag),
     (_flux_vegamag, u.Jy, _flux_jy)])
def test_flux_conversion_vega(in_q, out_u, ans):
    """Test Vega spectrum object and flux conversion with VEGAMAG.

    .. note:: 1% is good enough given Vega gets updated from time to time.

    """
    result = units.convert_flux(_wave, in_q, out_u, vegaspec=_vspec2)
    assert_quantity_allclose(result, ans, rtol=1e-2)

    # Scalar
    i = 0
    result = units.convert_flux(_wave[i], in_q[i], out_u, vegaspec=_vspec2)
    assert_quantity_allclose(result, ans[i], rtol=1e-2)


class TestSpectrumFromFile(test_spectrum_source.TestEmpiricalSourceFromFile):
    def setup_class(self):
        specfile = get_pkg_data_filename(
            os.path.join('data', 'hst_acs_hrc_f555w_x_grw70d5824.fits'),
            package='synphot.tests')
        self.sp = synphot_ext.FileSpectrum(specfile)

    def test_metadata(self):
        assert 'Spectrum' in str(self.sp)
        assert self.sp.meta['header']['SIMPLE']  # From FITS header
        assert self.sp.warnings == {}
        assert self.sp.z == 0
        assert_quantity_allclose(
            self.sp.waverange, [3479.99902344, 10500.00097656] * u.AA)
        
    def test_convert_wave(self):
        """Test wavelength conversion."""
        self.sp.convert('um')
        assert self.sp.waveunits == u.um
        assert np.allclose(self.sp.wave, self.sp.waveset.to_value(u.um))
        self.sp.convert(self.sp._internal_wave_unit)

    def test_convert_flux(self):
        """Test flux conversion."""
        self.sp.convert('flam')

        flam_units = self.sp._validate_flux_unit('flam')
        assert self.sp.fluxunits == flam_units

        flux_values = units.convert_flux(self.sp.wave, self.sp.flux, flam_units).value
        assert np.allclose(self.sp.flux, flux_values)

        self.sp.convert(self.sp._internal_flux_unit)

    def test_name_setter(self):
        """Test setting name."""
        self.sp.name = 'foo'
        assert self.sp.name == 'foo'
        assert self.sp.meta['expr'] == 'foo'

class TestBlackBodySource(test_spectrum_source.TestBlackBodySource):
    """Test source spectrum with BlackBody1D model."""
    def setup_class(self):
        self.sp = synphot_ext.Spectrum(BlackBodyNorm1D, temperature=5500)

class TestGaussianSource(test_spectrum_source.TestGaussianSource):
    """Test source spectrum with BlackBody1D model."""
    def setup_class(self):
        tf = 4.96611456e-12 * (u.erg / (u.cm * u.cm * u.s))
        self.sp = synphot_ext.Spectrum(GaussianFlux1D, total_flux=tf, mean=4000, fwhm=100)

def test_gaussian_source_watts():
    """https://github.com/spacetelescope/synphot_refactor/issues/153"""
    mu = 1 * u.um
    fwhm = (0.01 / 0.42466) * u.um
    flux = 1 * (u.W / u.m**2)

    sp = synphot_ext.Spectrum(GaussianFlux1D, mean=mu, fwhm=fwhm, total_flux=flux)
    tf = sp.integrate(flux_unit=units.FLAM)
    assert_quantity_allclose(tf, flux, rtol=1e-4)

class TestPowerLawSource(test_spectrum_source.TestPowerLawSource):
    """Test source spectrum with PowerLawFlux1D model."""
    def setup_class(self):
        self.sp = synphot_ext.Spectrum(PowerLawFlux1D, amplitude=1 * units.PHOTLAM,
                                 x_0=6000 * u.AA, alpha=4)
        self.w = np.arange(3000, 3100, 10) * u.AA

    @pytest.mark.xfail(reason='synphot_ext.Spectrum does create a default waveset')
    def test_no_default_wave(self):
        assert self.sp.waverange == [None, None]

        with pytest.raises(SynphotError, match='waveset is undefined'):
            self.sp(None)

def test_FlatSpectrum():
    sp = synphot_ext.Spectrum(ConstFlux1D, amplitude=1 * u.Jy)
    w = [1, 1000, 1e6] * u.AA
    with u.add_enabled_equivalencies(u.spectral_density(w)):
        assert_quantity_allclose(sp(w), 1 * u.Jy)
