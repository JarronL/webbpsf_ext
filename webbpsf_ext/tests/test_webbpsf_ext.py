import pytest

import numpy as np

import webbpsf_ext
from webbpsf_ext import synphot_ext
from webbpsf_ext.logging_utils import setup_logging
setup_logging(level='ERROR', verbose=False)

sp_vega = synphot_ext.Vega

def normalize_psf(arr, trim=3, max=False):
    norm_func = np.max if max else np.sum
    if (trim is None) or (trim == 0):
        return arr / norm_func(arr)
    else:
        return arr[trim:-trim,trim:-trim] / norm_func(arr[trim:-trim,trim:-trim])

def test_monochromatic(nrc_f335m_wext, nrc_f335m_webbpsf):
    """Test that webbpsf_ext and webbpsf give the same results for monochromatic PSFs"""

    nrc1 = nrc_f335m_wext
    nrc2 = nrc_f335m_webbpsf

    psf1 = nrc1.calc_psf(source=sp_vega, monochromatic=3.35e-6)
    psf2 = nrc2.calc_psf(source=sp_vega, monochromatic=3.35e-6, 
                         fov_pixels=nrc1.fov_pix, oversample=nrc1.oversample)

    assert np.allclose(psf1[0].data, psf2[0].data)

@pytest.mark.parametrize("filter", ['F323N', 'F335M', 'F356W'])
def test_load_psf_coeffs(filter, nrc_f335m_wext):
    """Test that PSF coefficients can be loaded"""

    nrc = nrc_f335m_wext
    nrc.filter = filter

    nrc.gen_psf_coeff()
    assert filter in nrc.save_name
    assert nrc.psf_coeff is not None

    nrc.gen_wfefield_coeff()
    assert nrc._psf_coeff_mod['si_field'] is not None

    nrc.gen_wfedrift_coeff()
    assert nrc._psf_coeff_mod['wfe_drift'] is not None

@pytest.mark.parametrize("xsci, ysci", [(1000,1000), (100,100), (2000,100), (100,2000), (2000,2000)])
def test_field_dependent_psfs(xsci, ysci, nrc_f335m_coeffs_cached, nrc_f335m_webbpsf):

    # WebbPSF PSF at 'sci' coordinates
    nrc0 = nrc_f335m_webbpsf
    nrc1 = nrc_f335m_coeffs_cached

    nrc0.detector_position = (xsci, ysci)
    fov_pix = nrc1.fov_pix
    osamp = nrc1.oversample
    psf0 = nrc0.calc_psf(source=sp_vega, fov_pixels=fov_pix, oversample=osamp)

    # WebbPSF Extended PSF at 'sci' coordinates, native and using coefficients
    psf1 = nrc1.calc_psf(sp=sp_vega, return_oversample=False,
                         coord_vals=(xsci, ysci), coord_frame='sci')
    psf2 = nrc1.calc_psf_from_coeff(sp=sp_vega, return_oversample=False,
                                   coord_vals=(xsci, ysci), coord_frame='sci')
    
    # Compare PSFs
    arr0 = normalize_psf(psf0[3].data)
    arr1 = normalize_psf(psf1[3].data)
    arr2 = normalize_psf(psf2[0].data)

    assert np.allclose(arr0, arr1, atol=0.001)
    assert np.allclose(arr1, arr2, atol=0.01)

@pytest.mark.parametrize("filter", ['F323N', 'F335M', 'F356W'])
def test_psfs_cached(nrc_f335m_webbpsf, nrc_f335m_wext, filter):

    # Create webbpsf and webbpsf_ext objects
    nrc0 = nrc_f335m_webbpsf
    nrc1 = nrc_f335m_wext

    # Set filter
    nrc0.filter = filter
    nrc1.filter = filter

    # Generate PSF coefficients
    nrc1.gen_psf_coeff()

    # Calculate PSFs with distortion and detector sampling
    fov_pix = nrc1.fov_pix
    osamp = nrc1.oversample
    psf0 = nrc0.calc_psf(source=sp_vega, fov_pixels=fov_pix, oversample=osamp)
    psf1 = nrc1.calc_psf(source=sp_vega, fov_pixels=fov_pix, oversample=osamp)
    psf2 = nrc1.calc_psf(sp=sp_vega, fov_pixels=fov_pix, oversample=osamp)
    psf3 = nrc1.calc_psf_from_coeff(sp=sp_vega, return_oversample=False)

    # Compare PSFs
    arr0 = normalize_psf(psf0[3].data)
    arr1 = normalize_psf(psf1[3].data)
    arr2 = normalize_psf(psf2[3].data)
    arr3 = normalize_psf(psf3[0].data)

    # Test webbpsf and webbpsf_ext PSFs using source=sp_vega
    # There will be a slight difference because the weights are not exactly the same
    assert np.allclose(arr1, arr0, atol=0.001)
    assert np.allclose(arr1, arr2, atol=0.0001)
    assert np.allclose(arr1, arr3, atol=0.0001)

@pytest.mark.parametrize("xidl, yidl", [(0,0), (0.1,0.1), (1,1)])
def test_coron_psfs(xidl, yidl, nrc_f335m_coeffs_cached):

    nrc = nrc_f335m_coeffs_cached

    # psf1 = nrc.calc_psf(sp=sp_vega)
    # psf2 = nrc.calc_psf_from_coeff(sp=sp_vega, return_oversample=False)

    psf1 = nrc.calc_psf(sp=sp_vega, coord_vals=(xidl,yidl), coord_frame='idl')
    psf2 = nrc.calc_psf_from_coeff(sp=sp_vega, return_oversample=False,
                                   coord_vals=(xidl,yidl), coord_frame='idl')

    # Compare PSFs
    arr1 = normalize_psf(psf1[3].data)
    arr2 = normalize_psf(psf2[0].data)
    assert np.allclose(arr1, arr2, atol=0.0001)
