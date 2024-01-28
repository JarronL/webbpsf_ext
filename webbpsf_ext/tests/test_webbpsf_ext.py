import pytest

import numpy as np

import webbpsf_ext
from webbpsf_ext import synphot_ext
from webbpsf_ext.logging_utils import setup_logging
setup_logging(level='ERROR', verbose=False)

sp_vega = synphot_ext.Vega

def test_monochromatic(nrc_f335m_wext, nrc_f335m_webbpsf):
    """Test that webbpsf_ext and webbpsf give the same results for monochromatic PSFs"""

    nrc1 = nrc_f335m_wext
    nrc2 = nrc_f335m_webbpsf

    psf1 = nrc1.calc_psf(source=sp_vega, monochromatic=3.35e-6)
    psf2 = nrc2.calc_psf(source=sp_vega, monochromatic=3.35e-6, 
                         fov_pixels=nrc1.fov_pix, oversample=nrc1.oversample)

    assert np.allclose(psf1[0].data, psf2[0].data)

@pytest.mark.parametrize("xsci, ysci", [(1000,1000), (100,100), (2000,100), (100,2000), (2000,2000)])
def test_field_dependent_psfs(xsci, ysci, nrc_f335m_coeffs, nrc_f335m_webbpsf):

    # WebbPSF PSF at 'sci' coordinates
    nrc0 = nrc_f335m_webbpsf
    nrc0.detector_position = (xsci, ysci)
    psf0 = nrc0.calc_psf(source=sp_vega, fov_pixels=33, oversample=2)

    # WebbPSF Extended PSF at 'sci' coordinates, native and using coefficients
    nrc = nrc_f335m_coeffs
    psf1 = nrc.calc_psf(sp=sp_vega, return_oversample=False,
                        coord_vals=(xsci, ysci), coord_frame='sci')
    psf2 = nrc.calc_psf_from_coeff(sp=sp_vega, return_oversample=False,
                                  coord_vals=(xsci, ysci), coord_frame='sci')
    
    # Compare PSFs
    trim = 3
    arr0 = psf0[3].data[trim:-trim,trim:-trim] / psf0[3].data[trim:-trim,trim:-trim].sum()
    arr1 = psf1[3].data[trim:-trim,trim:-trim] / psf1[3].data[trim:-trim,trim:-trim].sum()
    arr2 = psf2[0].data[trim:-trim,trim:-trim] / psf2[0].data[trim:-trim,trim:-trim].sum()

    assert np.allclose(arr0, arr1, atol=0.001)
    assert np.allclose(arr1, arr2, atol=0.01)

@pytest.mark.parametrize("filter", ['F323N', 'F335M', 'F356W'])
def test_psfs(nrc_f335m_webbpsf, nrc_f335m_wext, filter):

    # Create webbpsf and webbpsf_ext objects
    nrc0 = nrc_f335m_webbpsf
    nrc1 = nrc_f335m_wext

    # Set filter
    nrc0.filter = filter
    nrc1.filter = filter

    # Generate PSF coefficients
    nrc1.gen_psf_coeff(save=False, force=True)

    # Calculate PSFs with distortion and detector sampling
    fov_pix = nrc1.fov_pix
    osamp = nrc1.oversample
    psf0 = nrc0.calc_psf(source=sp_vega, fov_pixels=fov_pix, oversample=osamp)
    psf1 = nrc1.calc_psf(source=sp_vega, fov_pixels=fov_pix, oversample=osamp)
    psf2 = nrc1.calc_psf(sp=sp_vega, fov_pixels=fov_pix, oversample=osamp)
    psf3 = nrc1.calc_psf_from_coeff(sp=sp_vega, return_oversample=False)

    # Compare PSFs
    trim = 3
    arr0 = psf0[3].data[trim:-trim,trim:-trim] / psf0[3].data[trim:-trim,trim:-trim].sum()
    arr1 = psf1[3].data[trim:-trim,trim:-trim] / psf1[3].data[trim:-trim,trim:-trim].sum()
    arr2 = psf2[3].data[trim:-trim,trim:-trim] / psf2[3].data[trim:-trim,trim:-trim].sum()
    arr3 = psf3[0].data[trim:-trim,trim:-trim] / psf3[0].data[trim:-trim,trim:-trim].sum()

    # Test webbpsf and webbpsf_ext PSFs using source=sp_vega
    # There will be a slight difference because the weights are not exactly the same
    assert np.allclose(arr1, arr0, atol=0.001)
    assert np.allclose(arr1, arr2, atol=0.0001)
    assert np.allclose(arr1, arr3, atol=0.0001)

def test_nircam_auto_pixelscale():
    # This test now uses approximate equality in all the checks, to accomodate the fact that
    # NIRCam pixel scales are drawn directly from SIAF for the aperture, and thus vary for each detector/
    #
    # 1.5% variance accommodates the differences between the various NRC detectors in each channel
    close_enough = lambda a, b: np.isclose(a, b, rtol=0.015)

    nc = webbpsf_ext.NIRCam_ext()

    nc.filter='F200W'
    assert close_enough(nc.pixelscale,  nc._pixelscale_short)
    assert nc.channel == 'short'

    # auto switch to long
    nc.filter='F444W'
    assert close_enough(nc.pixelscale,  nc._pixelscale_long)
    assert nc.channel == 'long'

    # and it can switch back to short:
    nc.filter='F200W'
    assert close_enough(nc.pixelscale,  nc._pixelscale_short)
    assert nc.channel == 'short'

    nc.pixelscale = 0.0123  # user is allowed to set something custom
    nc.filter='F444W'
    assert nc.pixelscale == 0.0123  # and that persists & overrides the default switching.


    # back to standard scale
    nc.pixelscale = nc._pixelscale_long
    # switch short again
    nc.filter='F212N'
    assert close_enough(nc.pixelscale,  nc._pixelscale_short)
    assert nc.channel == 'short'

    # And test we can switch based on detector names too
    nc.detector ='NRCA5'
    assert close_enough(nc.pixelscale,  nc._pixelscale_long)
    assert nc.channel == 'long'

    nc.detector ='NRCB1'
    assert close_enough(nc.pixelscale,  nc._pixelscale_short)
    assert nc.channel == 'short'

    nc.detector ='NRCA3'
    assert close_enough(nc.pixelscale,  nc._pixelscale_short)
    assert nc.channel == 'short'


    nc.auto_channel = False
    # now we can switch filters and nothing else should change:
    nc.filter='F480M'
    assert close_enough(nc.pixelscale,  nc._pixelscale_short)
    assert nc.channel == 'short'

    # but changing the detector explicitly always updates pixelscale, regardless
    # of auto_channel being False

    nc.detector = 'NRCA5'
    assert close_enough(nc.pixelscale,  nc._pixelscale_long)
    assert nc.channel == 'long'
