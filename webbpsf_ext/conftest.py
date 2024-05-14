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


####################################################
# Direct Imaging Fixtures

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

    # Will regenerate coefficients
    nrc_f335m_wext.gen_psf_coeff(save=False, force=True)
    nrc_f335m_wext.gen_wfefield_coeff(save=False, force=True)
    nrc_f335m_wext.gen_wfedrift_coeff(save=False, force=True)
    
    return nrc_f335m_wext

@pytest.fixture(scope='session')
def nrc_f335m_coeffs_cached(nrc_f335m_wext):
    """
    NIRCam object with PSF coefficients cached
    """

    # Will load from test DATA directory
    nrc_f335m_wext.gen_psf_coeff()
    nrc_f335m_wext.gen_wfefield_coeff()
    nrc_f335m_wext.gen_wfedrift_coeff()

    return nrc_f335m_wext


####################################################
# Coronagraph Fixtures

@pytest.fixture(scope='session')
def nrc_coron_wext():
    """
    Return NIRCam LW direct imaging object
    """
    nrc = webbpsf_ext.NIRCam_ext(filter='F335M', 
                                 pupil_mask='CIRCLYOT', 
                                 image_mask='MASK335R')
    nrc.save_dir = DATA_PATH

    nrc.fov_pix = 65
    nrc.oversample = 1

    nrc.options['jitter'] = None
    nrc.options['charge_diffusion_sigma'] = 0
    nrc.options['add_ipc'] = False

    nrc.options['pupil_shift_x'] = 0
    nrc.options['pupil_shift_y'] = 0
    nrc.options['pupilt_rotation'] = -0.5

    nrc.detector_position = (641.1, 1675.2)
    
    return nrc

@pytest.fixture(scope='session')
def nrc_coron_coeffs_cached(nrc_coron_wext):
    """
    NIRCam object with PSF coefficients cached
    """
    nrc = nrc_coron_wext

    # Will load from test DATA directory
    nrc.gen_psf_coeff()
    nrc.gen_wfemask_coeff(large_grid=False)
    nrc.gen_wfedrift_coeff()

    return nrc
