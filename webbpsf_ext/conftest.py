try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}

try:
    from webbpsf_ext import __version__ as version
except ImportError:
    version = 'unknown'

import pytest
from webbpsf_ext import NIRCam_ext, MIRI_ext

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


# @pytest.fixture(scope='session')
# def nrc_m335r():
#     """
#     Return NIRCam LW coronagraphy object
#     """
#     return NIRCam_ext(filter='F335M', pupil_mask='CIRCLYOT', image_mask='MASKM335R')

# @pytest.fixture(scope='session')
# def miri_fqpm1140():
#     """
#     Return NIRCam long wavelength coronagraphy SIAF.
#     """    
#     return MIRI_ext(filter='F1140C', pupil_mask='MASKFQPM', image_mask='FQPM1140')