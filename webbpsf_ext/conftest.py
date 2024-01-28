try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}

try:
    from webbpsf_ext import __version__ as version
except ImportError:
    version = 'unknown'

import os
import pytest
import webbpsf, webbpsf_ext

from webbpsf_ext.logging_utils import setup_logging
setup_logging(level='ERROR', verbose=False)

DATA_PATH = os.path.join(webbpsf_ext.__path__[0], 'tests', 'data/')

# Uncomment and customize the following lines to add/remove entries
# from the list of packages for which version numbers are displayed
# when running the tests.
def pytest_configure():
    PYTEST_HEADER_MODULES.pop('Pandas', None)
    PYTEST_HEADER_MODULES.pop('h5py', None)
    PYTEST_HEADER_MODULES['astropy'] = 'astropy'
    PYTEST_HEADER_MODULES['webbpsf'] = 'webbpsf'
    PYTEST_HEADER_MODULES['pysiaf'] = 'pysiaf'
    TESTED_VERSIONS['webbpsf_ext'] = version



@pytest.fixture(scope='session')
def nrc_f335m_webbpsf():
    """
    Return NIRCam LW from webbpsf
    """
    nrc = webbpsf.NIRCam()
    nrc.filter = 'F335M'

    nrc.options['fov_pixels'] = 33
    nrc.options['oversample'] = 2

    # Set defaults similar to webbpsf_ext
    nrc.options['jitter'] = None
    nrc.options['charge_diffusion_sigma'] = 0
    nrc.options['add_ipc'] = False

    nrc.detector_position = (1024,1024)

    return nrc

@pytest.fixture(scope='session')
def nrc_f335m_wext():
    """
    Return NIRCam LW direct imaging object
    """
    nrc = webbpsf_ext.NIRCam_ext(filter='F335M')
    nrc.save_dir = DATA_PATH

    nrc.fov_pix = 33
    nrc.oversample = 2

    nrc.options['jitter'] = None
    nrc.options['charge_diffusion_sigma'] = 0
    nrc.options['add_ipc'] = False

    nrc.detector_position = (1024,1024)
    
    return nrc

@pytest.fixture(scope='session')
def nrc_f335m_coeffs(nrc_f335m_wext):
    """
    NIRCam object with PSF coefficients generated
    """
    nrc = nrc_f335m_wext
    nrc.gen_psf_coeff()
    nrc.gen_wfefield_coeff()
    nrc.gen_wfedrift_coeff()
    return nrc


# @pytest.fixture(scope='session')
# def nrc_m335r():
#     """
#     Return NIRCam LW coronagraphy object
#     """
#     nrc = webbpsf_ext.NIRCam_ext(filter='F335M', pupil_mask='CIRCLYOT', image_mask='MASKM335R')
#     nrc.options['jitter'] = 'gaussian'
#     nrc.options['jitter_sigma'] = 0.001
#     nrc.options['charge_diffusion_sigma'] = 0
#     nrc.options['add_ipc'] = False

#     return 

# @pytest.fixture(scope='session')
# def miri_fqpm1140():
#     """
#     Return NIRCam long wavelength coronagraphy SIAF.
#     """    
#     return MIRI_ext(filter='F1140C', pupil_mask='MASKFQPM', image_mask='FQPM1140')